import React from 'react';
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

class EmissionsModel extends React.Component {
  constructor() {
    super()
    this.state = {
      inputSquareFootage: 50000,
      predictedEmissions: null
    }
    this.run = this.run.bind(this)
    this.visualize = this.visualize.bind(this)
    this.handleChange = this.handleChange.bind(this)
    this.predictEmissions = this.predictEmissions.bind(this)
  }
  async getData() {
    const bldgData = await fetch(
      "https://data.cityofnewyork.us/resource/qb3v-bbre.json?largest_property_use_type=Multifamily Housing&$where=starts_with(bbl_10_digits, '1')&$select=dof_gross_floor_area_ft,total_ghg_emissions_metric,address_1_self_reported&$limit=20000"
    );
    const bldgDataJSON = await bldgData.json();
    //turning strings into numbers
    const cleanedData = bldgDataJSON
      .map((bldg) => ({
        emissions: Number(bldg.total_ghg_emissions_metric),
        squareFootage: Number(bldg.dof_gross_floor_area_ft),
      }))
      //filtering outlier cases, i.e. buildings over 800,000 sqft and more than 6,000 tons of ghg emissions/year
      .filter((bldg) => bldg.emissions != 0
      && bldg.emissions < 6000
      && bldg.emissions != undefined && bldg.squareFootage != 0
      && bldg.squareFootage < 800000
      && bldg.squareFootage != undefined);
      //assigning values to x and y axes
      const values = await cleanedData.map((d) => ({
        x: d.squareFootage,
        y: d.emissions,
      }));

    return cleanedData;
  }
  model;
  normInputs;
  normLabels;
  inputTensor;
  labelTensor;

  createModel() {
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1 }));

    // model.add(tf.layers.dense({units: 3}));
    // model.add(tf.layers.dense({units: 3}));
    // model.add(tf.layers.dense({units: 15, activation: 'sigmoid'}));

    // // Add an output layer
    model.add(tf.layers.dense({ units: 1 }));
    this.model = model
    return model;
  }
  convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any intermediate tensors.

    return tf.tidy(() => {

      tf.util.shuffle(data);

      const inputs = data.map((d) => d.squareFootage);
      const labels = data.map((d) => d.emissions);
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]).clipByValue(0, 900000);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]).clipByValue(0, 5000);

      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor
        .sub(inputMin)
        .div(inputMax.sub(inputMin));

      const normalizedLabels = labelTensor
        .sub(labelMin)
        .div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
        inputTensor,
        labelTensor,
      };
    });
  }
  async trainModel(model, inputs, labels) {
    this.model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
    });


    const batchSize = 32;
    const epochs = 10;
    //6300/32 = ~196 batches; 5 * 196 = ~984

    return await this.model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ["loss", "mse"],
        { height: 200, callbacks: ["onEpochEnd"] },
      ),
    });
  }
  testModel(model, inputData, normalizationData) {
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

    const [xs, preds] = tf.tidy(() => {
      const xs = tf.linspace(0, 1, 100);
      const preds = model.predict(xs.reshape([100, 1]));
      const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
      const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
      return { x: val, y: preds[i] };
    });

    const originalPoints = inputData.map((d) => ({
      x: d.squareFootage,
      y: d.emissions,
    }));

    tfvis.render.scatterplot(
      { name: "Model Predictions vs Original Data" },
      {
        values: [originalPoints, predictedPoints],
        series: ["original", "predicted"],
      },
      {
        xLabel: "Square Footage",
        yLabel: "Emissions",
        height: 300,
      }
    );
  }
  async run() {
    const data = await this.getData();
    const values = data.map((d) => ({
      x: d.squareFootage,
      y: d.emissions,
    }));

    const model = this.createModel();
    tfvis.show.modelSummary({ name: "Model Summary" }, model);

    const tensorData = this.convertToTensor(data);
    const { inputs, labels, inputMax,
      inputMin,
      labelMax,
      labelMin,
    inputTensor,
  labelTensor } = tensorData;
    this.normInputs = inputs;
    this.normLabels = labels;
    this.inputTensor = inputTensor;
    this.labelTensor = labelTensor;


    await this.trainModel(model, inputs, labels);
    console.log("Done Training");

    this.testModel(model, data, tensorData);
  }
  async visualize() {
    const data = await this.getData();
    const values = data.map((d) => ({
      x: d.squareFootage,
      y: d.emissions,
    }));
    tfvis.render.scatterplot(
      { name: "Square Footage v Emissions" },
      { values },
      {
        xLabel: "Square Footage",
        yLabel: "Emissions",
        height: 300,
        zoomToFit: false
      }
    );
  }
  handleChange(e) {
    console.log('i was called', e.target)
    this.setState({
      inputSquareFootage: e.target.value
    })
    e.preventDefault()
  }
  normalize(tensor, previousMin = null, previousMax = null) {
    const max = previousMax || tensor.max()
    const min = previousMin || tensor.min()
    const normalizedTensor = tensor.sub(min).div(max.sub(min))
    return {tensor: normalizedTensor, min, max}
  }
  denormalize(tensor, min, max) {
    const denormalizedTensor = tensor.mul(max.sub(min)).add(min)
    return denormalizedTensor
  }
  predictEmissions(e) {
    e.preventDefault()
    if(isNaN(this.state.inputSquareFootage)) {
      alert("Please enter a valid number")
    } else {
      tf.tidy(() => {
        const pred = Number(this.state.inputSquareFootage)
        const predTensor = tf.tensor([pred])
        const model = this.model
        const normPred = this.normalize(predTensor, this.inputTensor.min(), this.inputTensor.max())
        const prediction = model.predict(normPred.tensor)
        const label = this.denormalize(prediction, this.labelTensor.min(), this.labelTensor.max())
        let labelValue = label.dataSync()[0]
        this.setState({
          predictedEmissions: labelValue
        })
      })
    }
  }
  render() {
    return (
    <div>
      <button onClick={this.visualize}>
        Visualize Data
      </button>
      <button onClick={this.run}>
        Run Model
      </button>
      <form>
        <label>
          Square Footage:
          <input type="text" value={this.state.inputSquareFootage} onChange={this.handleChange}/>
          <button onClick={this.predictEmissions}>Predict Emissions</button>
        </label>
      </form>
    <p>Predicted Emissions: {this.state.predictedEmissions}</p>
    </div>
    )
  }
}

export default EmissionsModel
