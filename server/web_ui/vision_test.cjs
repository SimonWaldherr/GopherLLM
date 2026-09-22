const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const context = vm.createContext({});
vm.runInContext(fs.readFileSync(__dirname + '/vision.js', 'utf8'), context);
const V = context.GopherLLMVision;
const det = (label, confidence, x_min, y_min, x_max, y_max) => ({class_id: 0, label, confidence, box: {x_min, y_min, x_max, y_max}});

test('cropRect pads, enforces a minimum and stays inside the image', () => {
  assert.deepEqual({...V.cropRect({x_min: 100, y_min: 100, x_max: 200, y_max: 300}, 1000, 1000)}, {x: 85, y: 70, w: 130, h: 260});
  const corner = V.cropRect({x_min: 0, y_min: 0, x_max: 10, y_max: 10}, 50, 40);
  assert.deepEqual({...corner}, {x: 0, y: 0, w: 32, h: 32});
  const whole = V.cropRect({x_min: 0, y_min: 0, x_max: 640, y_max: 480}, 640, 480);
  assert.deepEqual({...whole}, {x: 0, y: 0, w: 640, h: 480});
});

test('describePosition names the third of the frame', () => {
  assert.equal(V.describePosition({x_min: 0, y_min: 0, x_max: 10, y_max: 10}, 300, 300), 'top left');
  assert.equal(V.describePosition({x_min: 140, y_min: 140, x_max: 160, y_max: 160}, 300, 300), 'centre');
  assert.equal(V.describePosition({x_min: 250, y_min: 140, x_max: 290, y_max: 160}, 300, 300), 'middle right');
});

test('describeSelection tells the model what it sees and that the detector can be wrong', () => {
  const one = V.describeSelection([det('person', 0.876, 0, 0, 10, 10)], 800, 600, 'upload');
  assert.match(one, /crop of a larger 800×600 image/);
  assert.match(one, /"person" \(88% confidence\), in the top left/);
  assert.match(one, /can be wrong/);
  const two = V.describeSelection([det('dog', 0.5, 0, 0, 10, 10), det('car', 0.91, 700, 500, 800, 600)], 800, 600, 'live');
  assert.match(two, /collage of 2 numbered crops from the current live frame/);
  assert.match(two, /1\. dog \(50%\), top left\n2\. car \(91%\), bottom right/);
  assert.equal(V.describeSelection([], 1, 1), '');
});

test('collageLayout packs crops into justified rows without square padding', () => {
  // Two tall crops share one 100px row: widths follow their aspect ratios.
  const two = V.collageLayout([{w: 50, h: 100}, {w: 100, h: 200}], 100, 1000, 4);
  assert.deepEqual([...two.tiles].map((t) => ({...t})), [{x: 0, y: 0, w: 50, h: 100}, {x: 54, y: 0, w: 50, h: 100}]);
  assert.equal(two.width, 104);
  assert.equal(two.height, 100);
  // A row too wide for maxWidth shrinks its height instead.
  const wide = V.collageLayout([{w: 400, h: 100}, {w: 400, h: 100}], 100, 404, 4);
  assert.equal(wide.tiles[0].h, 50);
  assert.equal(wide.width, 404);
  // Four crops become two rows of two.
  const four = V.collageLayout([{w: 1, h: 1}, {w: 1, h: 1}, {w: 1, h: 1}, {w: 1, h: 1}], 10, 1000, 2);
  assert.deepEqual({...four.tiles[3]}, {x: 12, y: 12, w: 10, h: 10});
  assert.equal(four.height, 22);
});

test('sceneChanged ignores jitter but notices new, missing and moved objects', () => {
  const before = [det('person', 0.9, 100, 100, 200, 300)];
  assert.equal(V.sceneChanged(null, before), true);
  assert.equal(V.sceneChanged(before, [det('person', 0.8, 105, 98, 204, 305)]), false);
  assert.equal(V.sceneChanged(before, [det('person', 0.8, 400, 100, 500, 300)]), true, 'moved');
  assert.equal(V.sceneChanged(before, [...before, det('person', 0.7, 400, 100, 500, 300)]), true, 'second person');
  assert.equal(V.sceneChanged(before, [det('dog', 0.9, 100, 100, 200, 300)]), true, 'different label');
});

test('parseClassList and strongest', () => {
  assert.deepEqual(Array.from(V.parseClassList(' Person, dog ,,CAR ')), ['person', 'dog', 'car']);
  assert.equal(V.parseClassList('').size, 0);
  const picked = V.strongest([det('a', 0.2, 0, 0, 1, 1), det('b', 0.9, 0, 0, 1, 1), det('c', 0.5, 0, 0, 1, 1)], 2);
  assert.deepEqual(picked.map((d) => d.label), ['b', 'c']);
});

test('scaleDetections maps boxes from the detector frame to the source', () => {
  const [d] = V.scaleDetections([det('x', 1, 10, 20, 30, 40)], 2);
  assert.deepEqual({...d.box}, {x_min: 20, y_min: 40, x_max: 60, y_max: 80});
});

test('detect posts the model, image and class filter', async () => {
  let seen;
  class FormData { constructor() { this.fields = []; } append(k, v) { this.fields.push([k, typeof v === 'string' ? v : 'blob']); } }
  context.FormData = FormData;
  const fetch = async (url, init) => { seen = {url, fields: init.body.fields}; return {ok: true, json: async () => ({detections: [det('person', 0.9, 0, 0, 1, 1)]})}; };
  const dets = await V.detect(fetch, 'yolo11n', {}, {classes: V.parseClassList('person,dog'), conf: 0.3});
  assert.equal(dets.length, 1);
  assert.equal(seen.url, '/v1/vision/detections');
  assert.deepEqual(seen.fields, [['model', 'yolo11n'], ['file', 'blob'], ['conf', '0.3'], ['classes', 'person,dog']]);
  const failing = async () => ({ok: false, text: async () => 'stock model is not downloaded\n'});
  await assert.rejects(V.detect(failing, 'yolo11n', {}), /not downloaded/);
});

const plain = (x) => JSON.parse(JSON.stringify(x));

test('countByLabel and summarizeDetections', () => {
  const dets = [det('person', 0.9, 0, 0, 1, 1), det('bicycle', 0.5, 0, 0, 1, 1), det('person', 0.4, 0, 0, 1, 1)];
  assert.deepEqual(plain(V.countByLabel(dets)), [['person', 2], ['bicycle', 1]]);
  assert.match(V.summarizeDetections(dets), /found in the attached image: 2 × person, 1 × bicycle\./);
  assert.match(V.summarizeDetections([]), /found no objects/);
});

test('createCountTracker reports only changes confirmed over consecutive frames', () => {
  const track = V.createCountTracker(2);
  const person = det('person', 0.9, 0, 0, 1, 1), dog = det('dog', 0.9, 0, 0, 1, 1);
  assert.deepEqual(plain(track([person])), [], 'first sighting is not yet confirmed');
  assert.deepEqual(plain(track([person])), [{label: 'person', from: 0, to: 1}]);
  assert.deepEqual(plain(track([person])), [], 'no change');
  assert.deepEqual(plain(track([person, dog])), [], 'one-frame flicker');
  assert.deepEqual(plain(track([person])), [], 'flicker gone again');
  assert.deepEqual(plain(track([person, person])), []);
  const events = track([person, person]);
  assert.deepEqual(plain(events.map(V.describeEvent)), ['person: 1 → 2']);
  track([]);
  assert.deepEqual(plain(track([])).map(V.describeEvent), ['person left']);
  assert.equal(V.describeEvent({label: 'car', from: 0, to: 3}), 'car appeared (3)');
});
