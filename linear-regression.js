const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

/**  @class */
class LinearRegression {
  /**  
  @param {number[][]} features
  @param {number[][]} labels
  @param {{learningRate?: number, iterations?: number, batchSize?: number}} options
 */
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];

    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000, batchSize: 10 },
      options
    );

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  _gradientDescentOld() {
    const currentGuessesForMpg = this.features.map((row) => {
      return this.m * row[0] + this.b;
    });

    const bSlope =
      (_.sum(
        currentGuessesForMpg.map((guess, i) => {
          return guess - this.labels[i][0];
        })
      ) *
        2) /
      this.features.length;

    const mSlope =
      (_.sum(
        currentGuessesForMpg.map((guess, i) => {
          return -1 * this.features[i][0] * (this.labels[i][0] - guess);
        })
      ) *
        2) /
      this.features.length;

    this.m -= mSlope * this.options.learningRate;
    this.b -= bSlope * this.options.learningRate;
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights);
    const diferences = currentGuesses.sub(labels);


    const slopes = features
      .transpose()
      .matMul(diferences)
      .div(features.shape[0]);
        
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  /**
   *
   * @param {'normal' | 'batch'} gradientType
   */
  train(gradientType = "normal") {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );

    for (let i = 0; i < this.options.iterations; i++) {
      switch (gradientType) {
        case "normal":
          this.gradientDescent(this.features, this.labels);
          break;
        case "batch":
          for (let j = 0; j < batchQuantity; j++) {
            const startIndex = j * this.options.batchSize;
            const { batchSize } = this.options;

            const featureSlice = this.features.slice(
              [startIndex, 0],
              [batchSize, -1]
            );
            const labelSlice = this.labels.slice(
              [startIndex, 0],
              [batchSize, -1]
            );


            this.gradientDescent(featureSlice, labelSlice);
          }

          break;
      }

      this.recordMSE();
      this.updateLearningRate();
    }
  }

  /**
   *
   * @param {any[]} testFeatures
   * @param {any[]} testLabels
   */


  predict(observations) {

    return this.processFeatures(observations).matMul(this.weights);
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);

    const res = testLabels.sub(predictions).pow(2).sum().arraySync();
    const tot = testLabels.sub(testLabels.mean()).pow(2).sum().arraySync();

    return 1 - res / tot;
  }

  /**
   *
   * @param {any[]} features
   * @returns {tf.Tensor}
   */
  processFeatures(features) {
    features = tf.tensor(features);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  /**
   *
   * @param {tf.Tensor} features
   */
  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance.add(1e-7);

    features = features.sub(mean).div(variance.add(1e-7).pow(0.5));

    return features;
  }

  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .arraySync();

    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) return;

    if (this.mseHistory[0] > this.mseHistory[1]) this.options.learningRate /= 2;
    else this.options.learningRate *= 1.05;
  }
}

module.exports = LinearRegression;
