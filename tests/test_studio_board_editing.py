"""Execute the real board-card duration handlers before an asynchronous redraw."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "studio" / "static" / "app.js"

JS = r"""
const fs = require('node:fs');
const assert = require('node:assert/strict');
const src = fs.readFileSync(process.argv[1], 'utf8');
const start = src.indexOf('function renderBoardCard(');
const end = src.indexOf('function bindBoardDrag(', start);
const renderSource = src.slice(start, end);
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.events = {}; this.dataset = {};
    this.style = {}; this.classList = {add(){}, remove(){}};
  }
  append(...els) { this.children.push(...els); }
  appendChild(el) { this.children.push(el); }
  add(el) { this.children.push(el); }
  addEventListener(name, fn) { (this.events[name] ||= []).push(fn); }
  fire(name) { for (const fn of this.events[name] || []) fn({}); }
}
const document = {createElement: tag => new Element(tag)};
const state = {boardBusy: {}, shotVideos: {}};
const boardShots = n => n.params.shots;
const boardShotVideo = s => s.video;
const boardMotions = () => [['static', 'fixed']];
const bindBoardDrag = () => {};
const Option = function() {};
const render = eval('(' + renderSource.trim() + ')');
const shot = {id:'s1', prompt:'scene', duration:4, motion:'static', variants:[]};
const node = {params:{aspect:'dome', shots:[shot]}};
function duration(card) {
  if (card.type === 'number') return card;
  for (const child of card.children) { const d = duration(child); if (d) return d; }
}
const input = duration(render(node, shot, 0, new Set()));
input.value = '1.5'; input.fire('input');
assert.equal(shot.duration, 1.5, 'duration must persist before blur/redraw');
assert.equal(duration(render(node, shot, 0, new Set())).value, 1.5);
assert.equal(input.value, '1.5', 'typing must not be rewritten');
input.value = ''; input.fire('input');
assert.equal(shot.duration, 1.5, 'empty draft must not overwrite the model');
input.value = 'invalid'; input.fire('input');
assert.equal(shot.duration, 1.5);
input.value = '10'; input.fire('input');
assert.equal(shot.duration, 10);
input.value = '20'; input.fire('input');
assert.equal(shot.duration, 10);
assert.equal(input.value, '20', 'clamp only normalizes the text on change');
input.fire('change'); assert.equal(input.value, 10);
input.value = ''; input.fire('change'); assert.equal(shot.duration, 4);
input.value = '-1'; input.fire('change'); assert.equal(shot.duration, 1);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node required for real JS event regression")
def test_duration_survives_redraw_before_blur() -> None:
    result = subprocess.run(
        [shutil.which("node"), "-e", JS, str(APP)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
