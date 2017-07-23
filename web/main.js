'use strict';

function loadJSON(path) {
  return new Promise((resolve, reject) => {
    let xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    xobj.open('GET', path, true);
    xobj.onreadystatechange = function() {
      if (xobj.readyState == 4 && xobj.status == "200") {
        resolve(JSON.parse(xobj.responseText));
      } else if (xobj.status != "200") {
        reject(xobj);
      }
    };
    xobj.send(null);
  });
};

function loadModelDict(model_name){
  return loadJSON('assets/models/'  + model_name + '/dict.json');
}

function loadModel(model_name) {
  const model = new KerasJS.Model({
    filepaths: {
      model: 'assets/models/' + model_name + '/model.json',
      weights: 'assets/models/' + model_name + '/model_weights.buf',
      metadata: 'assets/models/' + model_name + '/model_metadata.json'
    },
    gpu: true
  });

  return model.ready()
    .then(() => loadModelDict(model_name))
    .then(encode_dict => {

      // create decode dict
      const decode_dict = {};
      for (const key in encode_dict) {
        decode_dict[encode_dict[key]] = key;
      }

      const input_layer =  {
        name: model.data.model.config.input_layers[0][0],
        shape: model.modelLayersMap.get(model.data.model.config.input_layers[0][0]).shape
      };
      const output_layer = {
        name: model.data.model.config.output_layers[0][0]
      };

      return { model,
               input_layer,
               output_layer,
               encode_dict,
               decode_dict,
               encode: e => {
                 const res = {};
                 res[input_layer.name] = new Float32Array(
                   [].concat.apply([],
                                   e.split('').map(t => {
                                     const code = encode_dict[t];
                                     const res = new Array(Object
                                                           .keys(encode_dict)
                                                           .length + 1).fill(0);
                                     res[code] = 1;
                                     return res;
                                   })));
                 return res;
               },
               decode: (d, temperature=1.0) => {
                 d = Array.prototype.slice.call(new Float32Array(d.buffer));

                 let probas = d;
                 if(temperature != 1.0) {
                   probas = math.divide(math.log(probas), temperature);
                   probas = math.exp(probas);
                 }
                 probas = math.divide(probas, math.sum(probas));

                 const r = Math.random();

                 let acc = 0;
                 let selected = 0;
                 for (let i = 0; i < probas.length && acc < r; i++) {
                   acc += probas[i];
                   if (acc >= r) {
                     selected = i;
                   }
                 }
                 return decode_dict[selected];
               }
             };
    });
};

function generate(nn, base, iter, temp) {
  return new Promise((resolve, reject) => {
    if (iter <= 0) {
      resolve(base);
    } else {
      const res = nn.model.predict(nn.encode(base)).then(
        res => {
          base = base.substr(1);
          base = base + nn.decode(res[nn.output_layer.name], temp);
          generate(nn, base, iter - 1).then(resolve, reject);
        }, reject);
    }
  });
}

function complete(nn, text, nbChars, temp=1.0) {
  return generate(nn, text, nbChars, temp)
    .then(y => y.substring(text.length - nbChars));
}

function getNewCursorPos(textArea, startPos) {
  const txt = textArea.value;
  const nextSpace = txt.indexOf(' ', startPos);
  if(nextSpace > 0) {
    if(nextSpace == textArea.selectionStart) {
      // we are right before the space, jump over to end of next word
      return getNewCursorPos(textArea, nextSpace + 1);
    } else {
      // move cursor right before space
      return nextSpace;
    }
  } else {
    // we are at the last word, go at the end of text
    return txt.length;
  }
}

loadModel('default').then(nn => {
  const input = document.getElementById('user-input');
  const tempSlider = document.getElementById('temp-slider');
  const tempDisplay = document.getElementById('temp-display');
  tempDisplay.textContent = tempSlider.value;

  tempSlider.onchange = e => {
    // when slider change, update the displayed value and reset preds
    tempDisplay.textContent = tempSlider.value;
    input.value = input.value.substring(0, input.selectionStart);
  };

  let userEdit = false;

  input.onkeydown = e => {
    if(e.keyCode === 9) {
      // tab pressed, update cursor position
      input.selectionStart = getNewCursorPos(input, input.selectionStart);
      e.preventDefault();
    } else if (input.value[input.selectionStart] === e.key){
      // if the user is typing the next predicted char, just move the cursor
      input.selectionStart += 1;
      e.preventDefault();
    } else {
      // text updated, remove current prediction
      input.value = input.value.substring(0, input.selectionStart);
    }
  };

  function checkForUpdate() {
    const MIN_GEN = 40;

    if(input.value.length - input.selectionStart < MIN_GEN) {
      let x = input.value;
      let xlength = nn.input_layer.shape[0];

      if(x.length < xlength) {
        // pad begin of string with ' ' if needed
        x = new Array(xlength - x.length).fill(' ').join('') + x;
      }

      if(x.length > xlength) {
        // only keep the xlength last characters
        x = x.substring(x.length - xlength);
      }

      // get drunkness (aka temperature)
      const temperature = tempSlider.value;

      complete(nn, x, 1, temperature).then(text => {
        const caretPos = input.selectionStart;

        input.value = input.value + text;

        //restore caret position
        input.selectionStart = caretPos;
        input.selectionEnd = caretPos;
      });
    }
  }

  setInterval(() => {
    checkForUpdate();
  }, 100);

}, console.error);
