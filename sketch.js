UNIT_LENGTH_IN_PIXELS = 8;
const LEARNING_RATE = 0.4;

function setup() {
  createCanvas(innerWidth, innerHeight, WEBGL);
  rotateX(PI);

  config = createElement("table");
  config.addClass("config");
  usingConfig = false;
  config.attribute("onmousedown", "usingConfig = true;");

  graph = new Graph(terrain);
  x = 0;
  y = 0;
  sgd = new SGD(x, y);
  momentum = new Momentum(x, y);
  ada = new AdaGrad(x, y);
  rms = new RMSProp(x, y);
  adam = new Adam(x, y);
  reset();
  cam = createCamera();
  camera(0, height, height, 0,0,0, 0, 1, 0);
}

function draw() {
  background(0);
  if(!usingConfig) orbitControl();
  hoveringConfig = false;
  light(cam.eyeX,cam.eyeY,cam.eyeZ, 200, 200, 200);
  light(0,0, 1000, 100, 100, 100);
  coord.draw();
  graph.draw();
  sgd.update();
  sgd.draw();
  momentum.update();
  momentum.draw(255, 0, 0);
  ada.update();
  ada.draw(0, 255, 0);
  rms.update();
  rms.draw(0, 0, 255);
  adam.update();
  adam.draw(255, 255, 255);
  markCurrentBest();
  drawCursor();
}


class CoordinateSystem3D {
  constructor() {
  }

  draw() {
    push();
    stroke(255);
    let axis_lines = [
      [-width/2, 0, 0, width/2, 0, 0],
      [0, -height/2, 0, 0, height/2, 0],
      [0, 0, -width/2, 0, 0, width/2]
    ]
    let axis = 0;
    axis_lines.forEach(axis_line => {
      stroke(axis==0?255:0, axis==1?255:0, axis==2?255:0);
      line(...axis_line);
      push();
      noStroke();
      specularMaterial(axis==0?255:0, axis==1?255:0, axis==2?255:0);
      translate(...axis_line.slice(3));
      if(axis == 0) rotateZ(-PI/2);
      if(axis == 2) rotateX(PI/2);
      cone(10, 20, 10, 10, true);
      axis++;
      pop();
    });
    pop();
  }
}

class Graph {
  constructor(func, xlim=100, ylim=100, detailX=100, detailY=100) {
    this.func = func;
    let xUnit = xlim*2/detailX;
    let yUnit = ylim*2/detailY;
    this.geometry = new p5.Geometry(
      detailX, detailY,
      function() {
        for(let i = 0; i <= detailY; i++) {
          for(let j = 0; j<= detailX; j++) {
            let x = -xlim + (j*xUnit);
            let y = -ylim + (i*yUnit);
            this.vertices.push(new p5.Vector(x*UNIT_LENGTH_IN_PIXELS, y*UNIT_LENGTH_IN_PIXELS, func(x, y)*UNIT_LENGTH_IN_PIXELS))
          }
        }
        this.computeFaces();
        this.computeNormals();
        this.gid = `terrain-${random(100000)}`;
      }
    )
  }
  draw() {
    push();
    noStroke();
    ambientMaterial(255);
    model(this.geometry)
    pop();
  }
}


class Optimizer {
  path = [];
  r = 10;
  constructor(x, y, lr=LEARNING_RATE) {
    this.x = x;
    this.y = y;
    let tr = createElement("tr");
    this.isShown = createCheckbox(this.constructor.name, true);
    let td = createElement("td");
    td.child(this.isShown);
    let color = createElement("td");
    color.style("background-color", this.constructor.name == "SGD" ? "black" : this.constructor.name == "Momentum" ? "red" : this.constructor.name == "AdaGrad" ? "green" : this.constructor.name == "RMSProp" ? "blue" : "white");
    tr.child(td);
    tr.child(color);
    config.child(tr);
    this.lr = createSliderWithLabel("Learning Rate:", -1, 1, lr, 0.01);
  }

  reset(x, y) {
    this.x = x;
    this.y = y;
    this.z = terrain(x, y);
    this.path = [createVector(x, y, this.z)];
  }

  update() {
    if (this.isShown.checked()) this._update();
  }

  postUpdate() {
    this.z = terrain(this.x, this.y);
    // if the current point is only marginaly different to the last point we don't add it to the path
    if (p5.Vector.sub(this.path[this.path.length-1], createVector(this.x, this.y, this.z)).mag() < 0.5) return;
    this.path.push(createVector(this.x, this.y, this.z));
  }

  draw(r=0,g=0,b=0) {
    if (!this.isShown.checked()) return;
    push();
    stroke(r, g, b);
    strokeWeight(2);
    noFill();
    beginShape();
    this.path.forEach(p => {
      vertex(p.x*UNIT_LENGTH_IN_PIXELS, p.y*UNIT_LENGTH_IN_PIXELS, p.z*UNIT_LENGTH_IN_PIXELS+this.r);
    }
    )
    endShape();
    pop();
    push();
    noStroke();
    specularMaterial(r, g, b);
    translate(this.x*UNIT_LENGTH_IN_PIXELS, this.y*UNIT_LENGTH_IN_PIXELS, this.z*UNIT_LENGTH_IN_PIXELS+this.r);
    sphere(this.r);
    pop();
  }
}

class SGD extends Optimizer {
  constructor(x, y, lr=LEARNING_RATE) {
    super(x,y,lr);
    this.reset();
  }

  _update() {
    let d = gradient(this.x, this.y);
    this.x -= this.lr.value() * d.x;
    this.y -= this.lr.value() * d.y;
    this.postUpdate()
  }
}

class Momentum extends Optimizer {
  constructor(x, y, lr=LEARNING_RATE, momentum=0.9) {
    super(x,y,lr);
    this.momentum = createSliderWithLabel("Momentum:", 0, 1, momentum, 0.01);
    this.delta = createVector(0,0);
    this.reset();
  }

  reset(x, y) {
    super.reset(x, y);
    this.delta = createVector(0,0);
  }

  _update() {
    this.delta.mult(this.momentum.value()).add(gradient(this.x, this.y).mult(-this.lr.value()))
    this.x += this.delta.x;
    this.y += this.delta.y;
    this.postUpdate();
  }
}

class AdaGrad extends Optimizer {
  constructor(x, y, lr=LEARNING_RATE) {
    super(x,y,lr);
    this.sumOfGradientSquared = createVector(10e-7,10e-7);
    this.reset();
  }

  reset(x, y) {
    super.reset(x, y);
    this.sumOfGradientSquared = createVector(10e-7,10e-7);
  }

  _update() {
    let grad = gradient(this.x, this.y);
    this.sumOfGradientSquared.add(p5.Vector.mult(grad, grad));
    this.x -= this.lr.value() * grad.x / sqrt(this.sumOfGradientSquared.x);
    this.y -= this.lr.value() * grad.y / sqrt(this.sumOfGradientSquared.y);
    this.postUpdate();
  }
}

class RMSProp extends Optimizer {
  constructor(x, y, lr=LEARNING_RATE, decayRate=0.9) {
    super(x,y,lr);
    this.decayRate = createSliderWithLabel("Decay Rate:", 0, 1, decayRate, 0.01);
    this.reset();
  }

  reset(x, y) {
    super.reset(x, y);
    this.sumOfGradientSquared = createVector(10e-7,10e-7);
  }

  _update() {
    let grad = gradient(this.x, this.y);
    this.sumOfGradientSquared.mult(this.decayRate.value()).add(p5.Vector.mult(grad, grad).mult(1-this.decayRate.value()));
    this.x -= this.lr.value() * grad.x / sqrt(this.sumOfGradientSquared.x);
    this.y -= this.lr.value() * grad.y / sqrt(this.sumOfGradientSquared.y);
    this.postUpdate();
  }
}

class Adam extends Optimizer {
  constructor(x, y, lr=LEARNING_RATE, momentum=0.90, decayRate=0.9) {
    super(x,y,lr);
    this.momentum = createSliderWithLabel("Momentum:", 0, 1, momentum, 0.01)
    this.decayRate = createSliderWithLabel("Decay Rate:", 0, 1, decayRate, 0.01);
    this.reset();
  }

  reset(x, y) {
    super.reset(x, y);
    this.moment = createVector(0,0);
    this.sumOfGradientSquared = createVector(10e-7,10e-7);
    this.momentumProduct = this.momentum.value();
    this.decayRateProduct = this.decayRate.value();
  }

  _update() {
    let grad = gradient(this.x, this.y);
    // Update biased first moment estimate
    this.moment.mult(this.momentum.value()).add(p5.Vector.mult(grad, 1-this.momentum.value()))
    // Update biased second raw moment estimate
    this.sumOfGradientSquared.mult(this.decayRate.value()).add(p5.Vector.mult(grad, grad).mult(1-this.decayRate.value()));
    // Compute bias-corrected first moment estimate
    this.biasCorrectedMoment = p5.Vector.div(this.moment, 1-this.momentumProduct);
    this.momentumProduct *= this.momentum.value();
    // Compute bias-corrected second raw moment estimate
    this.biasCorrectedSumOfGradientSquared = p5.Vector.div(this.sumOfGradientSquared, 1-this.decayRateProduct);
    this.decayRateProduct *= this.decayRate.value();
    this.x -= this.lr.value() * this.biasCorrectedMoment.x / sqrt(this.biasCorrectedSumOfGradientSquared.x + 10e-8);
    this.y -= this.lr.value() * this.biasCorrectedMoment.y / sqrt(this.biasCorrectedSumOfGradientSquared.y + 10e-8);
    this.postUpdate();
  }
}

let coord = new CoordinateSystem3D();

function reset() {
  dragged = 5;
  sgd.reset(x, y);
  momentum.reset(x, y);
  ada.reset(x, y);
  rms.reset(x, y);
  adam.reset(x, y);
}



function drawCursor() {
  push();
  translate(getx() * UNIT_LENGTH_IN_PIXELS, gety() * UNIT_LENGTH_IN_PIXELS, terrain(getx(), gety())*UNIT_LENGTH_IN_PIXELS + 10);
  noFill();
  stroke(0, 0, 0);
  strokeWeight(5);
  circle(0, 0, 30);
  pop();
}

function markCurrentBest() {
  let contenders = [sgd, momentum, ada, rms, adam];
  let best = contenders.reduce((a, b) => a.z < b.z ? a : b);
  // we draw a circle around the best point
  push();
  noFill();
  stroke(255, 255, 0);
  strokeWeight(2);
  translate(best.x*UNIT_LENGTH_IN_PIXELS, best.y*UNIT_LENGTH_IN_PIXELS, best.z*UNIT_LENGTH_IN_PIXELS+best.r);
  circle(0, 0, 2*best.r + 20);
  pop();
}

function light(x,y,z,r=255,g=255,b=255) {
  pointLight(r, g, b, x, y, z);
}

function keyPressed() {
  if (keyCode === 32) {
    reset();
  } 
}

function mouseReleased() {
  if (dragged < 0 || usingConfig) {
    usingConfig = false;
    return;
  }
  x = getx();
  y = gety();
  reset();
}

getx = () => (cam.centerX + (mouseX-width/2) * (new p5.Vector(cam.eyeX, cam.eyeY, cam.eyeZ).mag()/800)) / UNIT_LENGTH_IN_PIXELS;
gety = () => (cam.centerY + (mouseY-height/2) * (new p5.Vector(cam.eyeX, cam.eyeY, cam.eyeZ).mag()/800)) / UNIT_LENGTH_IN_PIXELS;

function mouseClicked() {
  dragged = 5;
}

function mouseDragged() {
  dragged--;
}

function terrain(x, y) {
  //return (x**2 - (y*2)**2)/300
  return (noise((x/100+12341.1273), (y/100+128834.1287))-0.5)*100;
}

function gradient(x,y) {
  let dx = 0.01;
  let dy = 0.01;
  return createVector(
    (terrain(x+dx, y) - terrain(x-dx, y))/(2*dx),
    (terrain(x, y+dy) - terrain(x, y-dy))/(2*dy)
  )
}

function createSliderWithLabel(label, min, max, value, step) {
  let tr = createElement("tr");
  let labelElement = createElement("td");
  labelElement.html(label);
  let sliderOutput = createElement("td");
  sliderOutput.html(value.toFixed(2).slice(0,4));
  sliderOutput.id(`slider-output-${int(random(100000))}`);
  let slider = createSlider(min, max, value, step);
  slider.attribute("oninput", `document.getElementById('${sliderOutput.id()}').innerHTML = float(this.value).toFixed(2).slice(0,4);`);
  let sliderElement = createElement("td");
  sliderElement.child(slider);
  tr.child(labelElement);
  tr.child(sliderOutput);
  tr.child(sliderElement);
  config.child(tr);
  return slider;
}