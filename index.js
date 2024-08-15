require("@tensorflow/tfjs-node");
const loadCSV = require("./load-csv");
const LinearRegression = require("./linear-regression");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight" ,"displacement"],
  labelColumns: ["mpg"],
});

const learningRate = 0.1;
const iterations = 3;
const batchSize = 10;

const regression = new LinearRegression(features, labels, {
  learningRate,
  iterations,
  batchSize,
});

regression.train('batch');

const r2 = regression.test(testFeatures, testLabels);

plot({
  x: regression.mseHistory.reverse(),
  xLabel: "Iteration #",
  yLabel: "Mean Squared Error",
  name: "Batch"
});

console.log('-----------------------------------');
console.log()

console.log("[R2]: ", r2);


console.log('---------------------- PREDICTION ----------------------');


regression.predict([
  [ 130, 1.75, 307 ], // 18
  [ 150, 1.72, 304 ], // 16
  [ 95, 1.19, 113 ], // 24
]).print();


console.log('---------------------- PREDICTION ----------------------');

console.log()
console.log('-----------------------------------');
