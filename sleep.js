//https://enlight.nyc/neural-network/

const math = require('mathjs');
const fs = require('fs');

//X = (hours sleeping, hours studying, playing?)

//Training Data
this.X = [
  [1, 2, 9],
  [2, 1, 5],
  [5, 2, 9]
];
this.colMax = getColMax(this.X);
//console.log(this.colMax);

//y = score on test
this.y = [92, 86, 82];

//Data to be predicted
this.xp = [
  [1, 4, 8]
];

//scale units
this.X = scaleArray(this.X, this.colMax);
this.xp = scaleArray(this.xp, this.colMax)

//Incorrect Scaled Array
//this.xp = [[ 0.125,  0.5,  1.0 ]];

//max test score is 100
//this.y = matrix(this.y.map(x => x / 100));
this.y = math.matrix(this.y.map(x => x / 100));

/*

[[ 0.2         1.          1.        ]
 [ 0.4         0.5         0.55555556]
 [ 1.          1.          1.        ]]

 [ 1.  4.  8.]
xPredicted - Scaled
[ 0.125  0.5    1.   ] Not sure if this is correct

[ [ 0.2, 2, 0.8888888888888888 ] ]

*/

function getColMax(arr) {
  let result = [];
  arr = math.transpose(arr);
  for (let i = 0; i < arr.length; i++) {
    let row = arr[i];
    max = Math.max.apply(null, row);
    result.push(max);
  }
  return result;
}

function scaleArray(arr, colMax) {
  let result = [];
  for (let i = 0; i < arr.length; i++) {
    let row = arr[i];
    let scaledRow = [];
    for (let j = 0; j < row.length; j++) {
      scaledRow.push(row[j] / colMax[j]);
    }
    result.push(scaledRow);
  }
  return result;
}

function randomMatrix(rows, cols) {
  let newArray = [];
  for (let x = 0; x < rows; x++) {
    let row = [];
    for (let y = 0; y < cols; y++) {
      row.push(Math.random());
    }
    newArray.push(row);
  }
  return newArray;
}

function arrayAverage(arr) {
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum = sum + arr[i];
  }
  return sum / arr.length;
}

function sigmoid(s) {
  return 1 / (1 + Math.exp(-s));
}

//derivative of sigmoid
function sigmoidPrime(s) {
  return s * (1 - s);
}

class Neural_Network {

  constructor() {

    //Parameters
    this.inputSize = 3;
    this.outputSize = 1;
    this.hiddenSize = 3;

    //Weights
    //this.W1 = randomMatrix(this.inputSize, this.hiddenSize); //(3x2) weight matrix from input to hidden layer
    //this.W2 = randomMatrix(this.hiddenSize, this.outputSize); //(3x1) weight matrix from hidden to output layer

    this.W1 = math.matrix([
      [0.53014041, 0.11378556, -0.31211471],
      [0.43500565, 1.01304275, 0.52635022],
      [0.08742968, 1.77104636, -0.31388435],
    ])

    this.W2 = math.matrix([
      [0.93858602],
      [-1.09669252],
      [-0.16553269]
    ]);

  }

  //forward propagation through our network
  forward(X) {

    //dot product of X (input) and first set of 3x2 weights
    let z = math.multiply(X, this.W1);
    //console.log(z);

    /*
    [ [ 0.628463412, 2.806846222, 0.15004292800000002 ],
      [ 0.47813103344444446, 1.5359502434444445, -0.03605096844444447 ],
      [ 1.05257574, 2.89787467, -0.09964884000000002 ] ]
    */

    //activation function
    this.z2 = math.matrix(z.map(x => sigmoid(x)));

    /*
    [ [ 0.6521409639396806, 0.9430446617701235, 0.5374405171847592 ],
      [ 0.6173064485182823, 0.8228752384012125, 0.4909882338963319 ],
      [ 0.7412692051321124, 0.9477412742179805, 0.4751083841785858 ] ]
    */

    //dot product of hidden layer(z2) and second set of 3x1 weights
    //let z3 = this.z2.prod(this.W2);
    let z3 = math.multiply(this.z2,this.W2);

    /*
    [ [ -0.5111036092907004 ],
      [ -0.40432051932792595 ],
      [ -0.4222818222112498 ] ]
    */

    //final activation function
    let o = z3.map(x => sigmoid(x));

    /*
    [ 0.37493484940693095, 0.4002747327228629, 0.3959708592596488 ]
    */

    return o;
  }

  //backward propgate through the network
  backward(X, y, o) {

    //margin_of_error 
    let o_array = o._data; //convert the o matrix to an array for ease of use
    this.o_error = y.map(function (value, index, matrix) {
      return value - o_array[index];
    });

    /*
    [ 0.545065150593069, 0.4597252672771371, 0.42402914074035114 ]
    */

    //applying derivative of sigmoid to error

    //self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error
    let activations = o.map(x => sigmoidPrime(x));
    
    /*Expected activations: 
    [[ 0.23435871], [ 0.24005487], [ 0.23917794]]
    */

    let activationsArray = activations._data;
    this.o_delta = this.o_error.map(function (value, index, matrix) {
      return value * activationsArray[index];
    })

    //console.log(this.o_delta);

    /* Expected o_delta
    [ 0.12774076452721153, 0.11035928976222704, 0.10141841548195885 ]
    */

    //z2 error: how much our hidden layer weights contributed to output error
    //self.z2_error = self.o_delta.dot(self.W2.T)

    let wt = math.transpose(this.W2);
    let wtArray = math.flatten(wt)._data;
    console.log(wtArray);

    function dot(m1, m2) {
      let result = [];
      for (let i = 0; i < m1.length; i++) {
        let newRow = [];
        for (let j = 0; j < m2.length; j++) {
          let item = m1[i] * m2[j];
          newRow.push(item);
        }
        result.push(newRow);
      }
      return result;
    }

    let o_deltaArray = this.o_delta._data;
    this.z2_error = math.matrix(dot(o_deltaArray, wtArray));
    //console.log(this.z2_error;

    /* Expected z2_error

    [[ 0.1198957  -0.14009234 -0.02114527]
    [ 0.10358169 -0.12103021 -0.01826807]
    [ 0.09518991 -0.11122482 -0.01678806]]
    */

    //applying derivative of sigmoid to z2 error
    let t2 = this.z2.map(x => sigmoidPrime(x));
    //console.log(t2);

    /* Expected t2

    [[ 0.22685313  0.05371143  0.24859821]
    [ 0.2362392   0.14575158  0.24991879]
    [ 0.19178917  0.04952775  0.24938041]]
    */

    this.z2_delta = math.dotMultiply(this.z2_error, t2);
    //console.log(this.z2_delta);

    /* Expected z2_delta

    [[ 0.02719871 -0.00752456 -0.00525668]
    [ 0.02447005 -0.01764034 -0.00456553]
    [ 0.01825639 -0.00550872 -0.00418661]]
    */

    //adjusting first set (input --> hidden) weights
    let xt = math.transpose(X);
    
    let w1Adjustment = math.multiply(xt,this.z2_delta);
    //console.log(w1Adjustment);

    /* Expected w1Adjustment

    [[ 0.03348416 -0.01406976 -0.00706416]
    [ 0.05769013 -0.02185345 -0.01172606]
    [ 0.05904958 -0.02283347 -0.0119797 ]]
    */

    this.W1 = math.add(this.W1, w1Adjustment);
    //console.log(this.W1;

    /* Expected New W1

    [[ 0.56362457  0.0997158  -0.31917887]
    [ 0.49269578  0.9911893   0.51462416]
    [ 0.14647926  1.74821289 -0.32586405]]
    */

    //adjusting second set (hidden --> output) weights
    let z2t = math.transpose(this.z2);
    //console.log(z2t;

    let od = math.transpose(this.o_delta);
    //console.log(od());

    let w2Adjustment = math.transpose([math.multiply(z2t,od)]);
    //console.log(w2Adjustment);

    /* Expected w2Adjustment
    
    [[ 0.22660883]
    [ 0.30739559]
    [ 0.17102291]]
    */

    this.W2 = math.add(this.W2, w2Adjustment);

    /* Expected New W2

    [[ 1.16519485]
    [-0.78929693]
    [ 0.00549022]]
    */
  }

  train(X, y) {
    let o = this.forward(X);
    this.backward(X, y, o)
  }

  saveToFile() {
    let file = fs.createWriteStream('w1.txt');
    this.W1._data.forEach(function (v) {
      file.write(v.join(', ') + '\n');
    });
    file.end();

    file = fs.createWriteStream('w2.txt');
    this.W2._data.forEach(function (v) {
      file.write(v.join(', ') + '\n');
    });
    file.end();

    file = fs.createWriteStream('colMax.txt');
    this.W2().forEach(function (v) {
      file.write(v.join(', ') + '\n');
    });
    file.end();
  }

  predict(xp) {
    console.log("Predicted data based on trained weights: ");
    
    console.log("Input (scaled):");
    //xp2 = scaleArray2(xp);
    console.log(xp);
    console.log("");

    console.log("Output:");
    let result = this.forward(xp);
    console.log(result._data[0][0]);
    console.log("");
  }

}

NN = new Neural_Network();

//trains the NN 100,000 times
for (let i = 0; i < 100000; i++) {
  console.log(" #" + i + "\n")
  console.log("Input (scaled):");
  console.log(this.X);
  console.log("");
  console.log("Actual Output:");
  console.log(this.y._data);
  console.log("");
  console.log("Predicted Output:")
  let output = NN.forward(this.X);
  let outputArray = math.flatten(output._data);
  console.log(outputArray);

  console.log("");
  console.log("Loss:");

  this.error = this.y.map(function (value, index, matrix) {
    return value - outputArray[index];
  });
  //console.log(this.error);

  let e = math.square(this.error);
  eArray = e._data;
  //console.log(eArray);

  let loss = arrayAverage(eArray);
  console.log(loss);
  console.log("");

  //Expected Loss: 0.229414683987

  //console.log(np.mean(np.square(y - NN.forward(X))))) //mean sum squared loss
  NN.train(this.X, this.y)
  console.log("-----------------------------------------------------------");
}

//NN.saveToFile();
NN.predict(this.xp);

/*

Expected based on incorrect scaled array:

[[ 0.125,  0.5,  1.0 ]]
0.60040683

Expected based on correct scaled array:

[[ 0.2, 2, 0.8888888888888888 ]]
0.6497635182203974

*/
