/**
 * TTYREC player
 * @constructor
 * @param {object} term - Terminal to write TTY output to.
 * @param {object} opts - Config object with optional parameters.
 *    data - Arraybuffer with data of a TTYREC.
 *    start - Index of the frame to display.
 *    end - Index of the last frame to display.
 *    speed - Speed multiplier.  (i.e. speed=2 means to play twice as fast).
 *    frameDelay - Play every frame after a fixed delay.
 *    maxFrame - Cap the playback delay for a frame in milliseconds.
 *    err - Function for callback for errors.
 *
 * Depends upon an external TERM (i.e. https://github.com/chjj/term.js/).
 *
 * Usage:
 *   ttyPlay
 */

/* eslint-disable no-unused-vars */
// Disable lint warning for this function declaration (not used in this file).
const ttyPlay = function(term, opts) {
/* eslint-enable no-unused-vars */
  if (!term) throw Error('Must supply term.js terminal to TTYPlay');
  opts = opts || {};

  const clearScreen = '\033[2J'; // To clear terminal.

  let data = null;
  let frames = null;
  let timeout = null;
  let index = 0;
  let skipUntil = 0;
  let skipUntilAction = null;
  let speed = opts.speed || 1;
  let frameDelay = opts.frameDelay || 0;
  const maxFrame = opts.maxFrame || 0;
  const actionsMapping = opts.actionsMapping || {};
  const actionsDistribution = {};

  function parse(res, start, end) {
    start = start || 0;
    end = end || Number.MAX_SAFE_INTEGER;

    data = new DataView(res);
    frames = [];
    let parsedFrames = 0;

    let offset = 0;
    const size = data.byteLength;

    while (offset < size) {
      const sec = data.getUint32(offset, true);
      offset += 4;
      const usec = data.getUint32(offset, true);
      offset += 4;
      const length = data.getUint32(offset, true);
      offset += 4;
      const channel = data.getUint8(offset, true);
      offset += 1;

      parsedFrames += 1;

      if (parsedFrames > start && parsedFrames <= end + 2) {
        frames.push({
          time: sec * 1000 + usec / 1000,
          start: offset,
          length: length,
          channel: channel,
        });
        if (channel == 1) {
          const action = data.getUint8(offset);
          if (!(action in actionsDistribution)) {
            actionsDistribution[action] = 0;
          }
          actionsDistribution[action]++;
        }
      }

      offset += length;
    }
  }

  /** Display a frame or action.
   * Returns whether the playback should be stopped.
   */
  function step() {
    const current = frames[index];
    if (!current) return true;

    let stopAfterStep = false;

    // Convert ttyrec data to string.
    const str = String.fromCharCode.apply(
        null, new Uint8Array(data.buffer, current.start, current.length));
    if (current.channel == 0) {
      // Frame.
      term.write(str);
    } else {
      // Agent action.
      document.getElementById('last_action').innerText =
        `== Latest agent action ==\n` +
        `Charcode: ${str.charCodeAt(0)}\n` +
        `Meaning: ${actionsMapping[str.charCodeAt(0)]}`;
      if (skipUntilAction != null && str.charCodeAt(0) == skipUntilAction) {
        // Found action we had to skip until, stop skipping.
        skipUntilAction = null;
        // Stop playback.
        stopAfterStep = true;
      }
    }
    // Update frame count.
    document.getElementById('frame').innerText =
      `Frame: ${index} / ${frames.length - 1}`;

    index++;
    return stopAfterStep;
  }

  async function play(resolve=function() {}) {
    // Show a frame.
    const stopAfterStep = step();

    const current = frames[index-1];
    const next = frames[index];
    if (next && !stopAfterStep) {
      // Handle delay to next frame.
      let delay = next.time - current.time;
      delay = delay / speed;
      if (maxFrame != 0) {
        delay = delay > maxFrame ? maxFrame : delay;
      }
      if (frameDelay != 0) {
        delay = frameDelay;
      }
      if (index <= skipUntil || skipUntilAction != null) {
        // Frames are being skipped, play with no delay.
        play(resolve);
      } else {
        timeout = window.setTimeout(play, delay, resolve);
      }
    } else {
      // The promise is resolved only when the playback for
      // that episode is over (i.e. no next).
      resolve();
    }
  }

  function stop() {
    // Stop playback.
    window.clearTimeout(timeout);
  }

  function setSpeed(s) {
    speed = s;
  }

  function setFrameDelay(d) {
    frameDelay = d;
  }

  function jumpTo(frame) {
    if (frame < 0 || frame > frames.length) {
      console.log(`WARNING: trying to jump to invalid frame ${frame}, ` +
                  `in range 0-${frames.length - 1}.`);
      return;
    }
    stop();

    // Possible cases:
    // - frame >= index:
    //     just forward until that frame.
    // - frame < index:
    //     go back to the first frame, and then forward until the desired
    //     frame.
    skipUntil = frame;
    if (frame < index) {
      index = 0;
    }

    // Play forwards until the desired frame without setting timeouts.
    play();
    // As soon as a timeout is set (finished skipping), stop clears it
    // and we are wating on the desired frame.
    stop();
  }

  function seekAction(actionCharCode) {
    stop();
    skipUntilAction = actionCharCode;
    play();
  }

  function getActionsDistribution() {
    return actionsDistribution;
  }

  function close() {
    stop();
    // This should stop play, regradless wether there is a timeout
    // set or not.
    index = Number.MAX_SAFE_INTEGER - 1;
    term.write(clearScreen);
  }

  function reset() {
    stop();
    term.write(clearScreen);
    index = 0;
    play();
  }

  if (opts.data) {
    // Data is an arraybuffer contaning the ttyrec binary data.
    parse(opts.data, opts.start, opts.end);
  }

  return {
    parse: parse,
    play: play,
    step: step,
    stop: stop,
    close: close,
    reset: reset,
    setSpeed: setSpeed,
    setFrameDelay: setFrameDelay,
    jumpTo: jumpTo,
    seekAction: seekAction,
    getActionsDistribution: getActionsDistribution,
  };
};
