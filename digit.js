//https://enlight.nyc/neural-network/

const math = require('mathjs');

this.zero = [
    0, 1, 1, 0,
    1, 0, 0, 1,
    1, 0, 0, 1,
    1, 0, 0, 1,
    0, 1, 1, 0
]

this.one = [
    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 0, 1, 0
]

this.two = [
    0, 1, 1, 0,
    1, 0, 0, 1,
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 1, 1, 1
];

this.three = [
    1, 1, 1, 1,
    0, 0, 0, 1,
    0, 1, 1, 1,
    0, 0, 0, 1,
    1, 1, 1, 1
];

this.four = [
    0, 0, 1, 0,
    0, 1, 1, 0,
    1, 0, 1, 0,
    1, 1, 1, 1,
    0, 0, 1, 0
];

this.fourAlt1 = [
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 1, 0,
    1, 1, 1, 1,
    0, 0, 1, 0
];

this.five = [
    1, 1, 1, 1,
    1, 0, 0, 0,
    1, 1, 1, 0,
    0, 0, 0, 1,
    1, 1, 1, 0
];

this.six = [
    0, 1, 1, 1,
    1, 0, 0, 0,
    1, 1, 1, 0,
    1, 0, 0, 1,
    0, 1, 1, 0
];

this.seven = [
    1, 1, 1, 1,
    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 1, 0, 0,
    0, 1, 0, 0
];

this.eight = [
    0, 1, 1, 0,
    1, 0, 0, 1,
    0, 1, 1, 0,
    1, 0, 0, 1,
    0, 1, 1, 0
];

this.nine = [
    0, 1, 1, 1,
    1, 0, 0, 1,
    0, 1, 1, 1,
    0, 0, 0, 1,
    0, 0, 0, 1
];

/*

//2
this.predict = [
    0, 1, 1, 0,
    1, 0, 0, 1,
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 1, 1, 1
];

//4
this.predict = [
    0, 0, 1, 0,
    0, 1, 1, 0,
    1, 0, 1, 0,
    1, 1, 1, 1,
    0, 0, 1, 0
];

this.predict = [ //fourAlt1
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 1, 0,
    1, 1, 1, 1,
    0, 0, 1, 0
];

//5 - slightly adjusted - result after 10000: 4.891929752257355
this.predict = [
    1, 1, 1, 0,
    1, 0, 0, 0,
    1, 1, 1, 0,
    0, 0, 1, 0,
    1, 1 ,1, 0
];

//7 - slightly adjusted - result after 10000: 4.891929752257355
this.predict = [
    1, 1, 1, 1,
    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 0, 0, 0
];

*/

//4 - slightly different
this.predict = [
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 0, 1, 0,
    1, 1, 1, 1,
    0, 0, 1, 0
];

this.X = math.matrix([
this.zero, 
this.one, 
this.two, 
this.three, 
this.four, 
this.fourAlt1, 
this.five, 
this.six, 
this.seven, 
this.eight, 
this.nine]);

this.y = math.matrix([
    [0],
    [1],
    [2],
    [3],
    [4],[4],
    [5],
    [6],
    [7],
    [8],
    [9]
]);

let potentialNumbers = [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9];
this.y = math.matrix(potentialNumbers);

this.xPredicted = math.matrix(this.predict);

//Number of items in potentialNumbers (excluding zero)
var adjust = potentialNumbers.length - 1;

//scale units

//max test score is 100
this.y = math.matrix(this.y.map(x => x / adjust));

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

        //parameters
        this.inputSize = 20
        this.outputSize = 1
        this.hiddenSize = 20

        //weights
        this.W1 = randomMatrix(this.inputSize, this.hiddenSize); //(20x20) weight matrix from input to hidden layer
        this.W2 = randomMatrix(this.hiddenSize, this.outputSize); //(20x1) weight matrix from hidden to output layer
    }

    //forward propagation through our network
    forward(X) {

        //dot product of X (input) and first set of weights
        let z = math.multiply(X, this.W1);

        //activation function
        this.z2 = math.matrix(z.map(x => sigmoid(x)));

        //dot product of hidden layer (z2) and second set of weights
        let z3 = math.multiply(this.z2, this.W2);

        //final activation function
        let o = z3.map(x => sigmoid(x));

        return o;
    }

    //backward propgate through the network
    backward(X, y, o) {

        /*
        def backward(self, X, y, o): 
        self.o_error = y - o# error in output

        self.o_delta = self.o_error * self.sigmoidPrime(o)# applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T)# z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)# applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta)# adjusting first set(input-- > hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta)# adjusting second set(hidden-- > output) weights
        */

        //margin_of_error 
        let o_array = math.flatten(o)._data; //convert the o matrix to an array for ease of use
        this.o_error = y.map(function (value, index, matrix) {
            return value - o_array[index];
        });

        let activations = o.map(x => sigmoidPrime(x));

        let activationsArray = math.flatten(activations)._data;
        this.o_delta = this.o_error.map(function (value, index, matrix) {
            return value * activationsArray[index];
        })

        let wt = math.transpose(this.W2);
        let wtArray = math.flatten(wt);
        //console.log(wtArray);

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

        //applying derivative of sigmoid to z2 error
        let t2 = this.z2.map(x => sigmoidPrime(x));

        this.z2_delta = math.dotMultiply(this.z2_error, t2);

        //adjusting first set (input --> hidden) weights
        let xt = math.transpose(X);

        let w1Adjustment = math.multiply(xt, this.z2_delta);

        this.W1 = math.add(this.W1, w1Adjustment);

        //adjusting second set (hidden --> output) weights
        let z2t = math.transpose(this.z2);

        let od = math.transpose(this.o_delta);

        let w2Adjustment = math.transpose([math.multiply(z2t,od)]);

        this.W2 = math.add(this.W2, w2Adjustment);
    }

    train(X, y) {
        let o = this.forward(X);
        this.backward(X, y, o)
    }

    predict(xp) {

        /*
            print("Predicted data based on trained weights: ")
            print("Input (scaled): \n" + str(xPredicted))
            print("Actual Output: \n" + str((self.forward(xPredicted)) * adjust))
            print("Rounded Output: \n" + str(round((self.forward(xPredicted)) * adjust)))
        */

        console.log("Predicted data based on trained weights: ");

        console.log("Input (scaled):");
        console.log(xp._data);
        console.log("");

        console.log("Actual Output:");
        let output = this.forward(xp);
        output = output.map(x => x*adjust);
        console.log(output._data);
        console.log("");

        console.log("Rounded Output:");
        console.log(Math.round(output._data));

    }
}

NN = new Neural_Network();

//trains the NN 10,000 times
for (let i = 0; i < 10000; i++) {

    /*
  print("#" + str(i) + "\n")
  print("Input: \n" + str(X))
  print("Actual Output: \n" + str(y*adjust))
  print("Predicted Output: \n" + str(NN.forward(X)*adjust))
  print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  print("\n")
  NN.train(X, y)
  */
    console.log(" #" + i + "\n")
    console.log("Input (scaled):");
    console.log(this.X._data);
    console.log("");
    console.log("Actual Output:");
    let yOutput = this.y.map(x => x*adjust);
    console.log(yOutput._data);
    console.log("");
    console.log("Predicted Output:")
    let output = NN.forward(this.X);
    output = output.map(x => x*adjust);
    let outputArray = math.flatten(output._data);
    console.log(outputArray);

    console.log("");
    console.log("Loss:");

    this.error = this.y.map(function (value, index, matrix) {
        //console.log(value + " - " + outputArray[index] + ' = ' + (value - outputArray[index]));
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

NN.predict(this.xPredicted);
