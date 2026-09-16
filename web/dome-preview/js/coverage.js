/* coverage.js — measure how far a domemaster frame's content actually reaches.
 *
 * domemaster semantics assumed (see README):
 *   the inscribed circle of the square frame is the 180 deg hemisphere,
 *   circle centre = zenith, circle edge = horizon,
 *   normalised radius r maps LINEARLY to zenith angle:  theta = r * 90 deg.
 *
 * The measurement that matters: the largest r that still carries content.
 * A master whose content stops at r = 0.6 only fills 54 deg of zenith angle
 * and wastes the whole outer ring of the dome.
 */
(function (global) {
  'use strict';

  var SIZE = 512;        // analysis resolution (downsampled, plenty for this)
  var BINS = 256;        // radial histogram resolution
  var FILL_THRESHOLD = 0.5;   // a radius "has content" when >= 50% of its ring does
  var OUTER_LO = 0.8;    // the ring the owner got burned by
  var OUTER_HI = 0.98;

  var canvas = null;
  var ctx = null;

  function ensureCanvas() {
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.width = SIZE;
      canvas.height = SIZE;
      ctx = canvas.getContext('2d', { willReadFrequently: true });
    }
    return ctx;
  }

  /**
   * @param {ImageBitmap|HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} source
   * @param {number} lumaThreshold 0-255, a pixel counts as content above this
   * @returns {object|null} stats
   */
  function analyze(source, lumaThreshold) {
    var c = ensureCanvas();
    var thr = typeof lumaThreshold === 'number' ? lumaThreshold : 8;

    c.clearRect(0, 0, SIZE, SIZE);
    try {
      c.drawImage(source, 0, 0, SIZE, SIZE);
    } catch (e) {
      return null;
    }

    var data;
    try {
      data = c.getImageData(0, 0, SIZE, SIZE).data;
    } catch (e) {
      // Tainted canvas: happens on some browsers when the page itself is on
      // file:// . The README tells the user to serve over http in that case.
      return { error: 'tainted' };
    }

    var total = new Float64Array(BINS);
    var content = new Float64Array(BINS);
    var totalIn = 0, contentIn = 0;
    var outerTotal = 0, outerContent = 0;

    for (var y = 0; y < SIZE; y++) {
      var dy = ((y + 0.5) / SIZE) * 2 - 1;
      for (var x = 0; x < SIZE; x++) {
        var dx = ((x + 0.5) / SIZE) * 2 - 1;
        var rr = Math.sqrt(dx * dx + dy * dy);
        if (rr > 1) continue;
        var i = (y * SIZE + x) * 4;
        var a = data[i + 3];
        var lum = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
        var has = a > 16 && lum >= thr ? 1 : 0;

        var bin = Math.min(BINS - 1, (rr * BINS) | 0);
        total[bin] += 1;
        content[bin] += has;
        totalIn += 1;
        contentIn += has;
        if (rr >= OUTER_LO && rr <= OUTER_HI) {
          outerTotal += 1;
          outerContent += has;
        }
      }
    }

    var fill = new Float64Array(BINS);
    for (var b = 0; b < BINS; b++) {
      fill[b] = total[b] > 0 ? content[b] / total[b] : 0;
    }

    // Coverage radius: walk inwards from the rim, first radius whose ring is
    // at least half filled (and whose neighbour agrees, to shrug off noise).
    var covBin = -1;
    for (var k = BINS - 1; k >= 1; k--) {
      if (fill[k] >= FILL_THRESHOLD && fill[k - 1] >= FILL_THRESHOLD) {
        covBin = k;
        break;
      }
    }
    var coverageRadius = covBin < 0 ? 0 : (covBin + 1) / BINS;
    var coverageDeg = coverageRadius * 90;
    var theta = (coverageDeg * Math.PI) / 180;

    return {
      fill: fill,
      bins: BINS,
      coverageRadius: coverageRadius,
      coverageDeg: coverageDeg,
      // fraction of the hemisphere's solid angle actually lit: (1-cos theta)
      solidAngleFrac: 1 - Math.cos(theta),
      outerFill: outerTotal > 0 ? outerContent / outerTotal : 0,
      contentFrac: totalIn > 0 ? contentIn / totalIn : 0,
      outerLo: OUTER_LO,
      outerHi: OUTER_HI
    };
  }

  function verdict(stats) {
    if (!stats || stats.error) return { level: 'bad', text: '无法测量当前帧。' };
    var d = stats.coverageDeg;
    if (d >= 85) {
      return {
        level: 'ok',
        text: '合规：内容铺满整圆，覆盖到地平线附近，可直接出片。'
      };
    }
    if (d >= 75) {
      return {
        level: 'warn',
        text: '轻微留白：外圈最后几度没有内容，观众抬头看边缘会见到黑边。'
      };
    }
    return {
      level: 'bad',
      text:
        '不合规：内容只铺到天顶角 ' + d.toFixed(1) + '°（r=' +
        stats.coverageRadius.toFixed(2) + '），外圈大片留白。' +
        '出片前必须重做投影映射或扩幅，否则球幕四周是黑的。'
    };
  }

  global.Coverage = {
    analyze: analyze,
    verdict: verdict,
    SIZE: SIZE,
    BINS: BINS
  };
})(window);
