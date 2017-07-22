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
      weights: 'assets/models/' + model_name + '/weights.buf',
      metadata: 'assets/models/' + model_name + '/metadata.json'
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

      return { model,
               encode_dict,
               decode_dict,
               encode: e => ({
                 'input': new Float32Array(
                   [].concat.apply([],
                                   e.split('')
                                   .map(t => {
                                     const code = encode_dict[t];
                                     const res = new Array(Object
                                                           .keys(encode_dict)
                                                           .length).fill(0);
                                     res[code] = 1;
                                     return res;
                                   })))
               }),
               decode: d => {
                 // find max value
                 let max = 0;
                 let max_i = 0;
                 for (let i = 0; i < d.length; i++) {
                   if (d[i] > max) {
                     max_i = i;
                     max = d[i];
                   }
                 }
                 return decode_dict[max_i];
               }
             };
    });
};

function generate(nn, base, iter) {
  return new Promise((resolve, reject) => {
    if (iter <= 0) {
      resolve(base);
    } else {
      const res = nn.model.predict(nn.encode(base)).then(
        res => {
          base = base.substr(1);
          base = base + nn.decode(res.output);
          generate(nn, base, iter - 1).then(resolve, reject);
        }, reject);
    }
  });
}

loadModel('default').then(nn => {
  const x = 'obligations mises à la charge des opér';
  generate(nn, x, 20).then(console.log);
}, console.error);
