const brain = require('brain.js');
const fs = require('fs');

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

//4 - slightly different
this.predict = [
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 0, 1, 0,
    1, 1, 1, 1,
    0, 0, 1, 0
];
*/

//7 - slightly adjusted - result after 10000: 4.891929752257355
this.predict = [
    1, 1, 1, 1,
    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 0, 0, 0
];

var net = new brain.NeuralNetwork();

var update = function (result) {

    /*
    {
        error: 0.0039139985510105032,  // training error
        iterations: 406                // training iterations
    }
    */

    //console.log(result);
}

var options = {
    // Defaults values --> expected validation
    iterations: 100000,   // the maximum times to iterate the training data --> number greater than 0
    errorThresh: 0.001,   // the acceptable error percentage from training data --> number between 0 and 1
    log: true,            // true to use console.log, when a function is supplied it is used --> Either true or a function
    logPeriod: 10,        // iterations between logging out --> number greater than 0
    learningRate: 0.3,    // scales with delta to effect training rate --> number between 0 and 1
    momentum: 0.1,        // scales with next layer's change value --> number between 0 and 1
    callback: update,     // a periodic call back that can be triggered while training --> null or function
    callbackPeriod: 5,    // the number of iterations through the training data between callback calls --> number greater than 0
    timeout: Infinity     // the max number of milliseconds to train for --> number greater than 0
}

/*
net.train(
    [
        { input: this.zero, output: { 0: 1 } },
        { input: this.one, output: { 1: 1 } },
        { input: this.two, output: { 2: 1 } },
        { input: this.three, output: { 3: 1 } },
        { input: this.four, output: { 4: 1 } },
        { input: this.fourAlt1, output: { 4: 1 } },
        { input: this.five, output: { 5: 1 } },
        { input: this.six, output: { 6: 1 } },
        { input: this.seven, output: { 7: 1 } },
        { input: this.eight, output: { 8: 1 } },
        { input: this.nine, output: { 9: 1 } }

    ],
    options
);

//Write the training result to a file
fs.writeFileSync('netdata.json',JSON.stringify(net.toJSON()));

*/

//Load the data from the json file
var netdata = JSON.parse(fs.readFileSync('netdata.json', 'utf8'));
net.fromJSON(netdata);

var output = net.run(this.predict);  // { white: 0.99, black: 0.002 }
console.log(output);

var result = getResult(output);

console.log('-------------------------------------');
console.log('Expected char is:');

console.log(result);

function getResult(obj) {
    var arr = Object.keys(obj).map(function (key) { return obj[key]; });
    var max = Math.max.apply(null, arr);
    for (p in obj) {
        if (obj[p] === max) {
            var result = {value: p.toString(), probability: max};
            return result;
            break;
        }
    }
}
