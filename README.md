Quadra_neural
=============

Write your own press releases in the style of [La Quadrature du Net](https://www.laquadrature.net/) with the power of neural networks.

This website creates a text area that autofills based on the content typed by the user with text generated by a neural network.
The generation process is done client-side thanks to [Keras.js](https://github.com/transcranial/keras-js).

A slider allows to control the "drunkenness" of the model, aka how "random" the generation process it. A low value means the process will select the most common character nearly every times, while a high value allows for more variation in the selection (which leads to more diversity but also to a less realistic text).

This project contains all the necessary code to create an adequate model from scratch. It also contain a version of the model binaries generated using the `model_gen.ipynb` notebook, stored in a separate `with-model` branch. This branch is used specifically as a quick way to distribute the binaries, and can be rebased over without notice, so be careful not to build over it!

If you want to use the provided model, simply checkout the `with-model` branch and you are good to go. If you want to create your own model, see the "Generating a model" section of this README.

A live demo is available at [http://jorquera.net/quadra\_neural](http://jorquera.net/quadra_neural).

## Content

This project is organized in two parts:

- `web` contains the code neural-network powered web editor
- `generation` contains the necessary code to generate a compatible model

## Generating a model

- Go into the `generation` folder
- Run the `quadra_scrape.py` - This script should download and convert all the press releases from LQDN into a `data` folder
- The `model_gen.ipynb` contains the code to generate a model based  the content of `data`. If you are not interested in looking into the notebook for the process details, you can run it from a terminal with `jupyter nbconvert --execute model_gen.ipynb`
- By default, the notebook produces several folders named `model_<date>` corresponding to the different training stages. Select the one corresponding to the version you want to use (usually the last one) and convert it in Keras.js format with the [Keras.js encoder.py script](https://github.com/transcranial/keras-js/blob/master/encoder.py).
- Create the `web/assets/models/default` folder and copy the model files into it (the `dict.json`, `model.json`, `model_metadata.json` and `model_weights.buf`. You can actually skip the `model.hdf5` file).
- You're all set! Generate away

When generating your own model, remember that the decoding is done client-side. If you go overboard with your model's architecture the files to download by the browser will grow fast, and if the decoding takes too much time on the user machine the web editor will become unresponsive.

## License

All the content of this project is licensed under [the AGPLv3 license](https://www.gnu.org/licenses/agpl-3.0.en.html)
