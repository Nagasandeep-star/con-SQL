/**
 * LatticeLoader component from React Bits
 * UMD & Browser-ready implementation compatible with React 17/18/19
 * Also exposes vanilla DOM factory createLatticeLoader for universal embedding.
 */

(function (global, factory) {
  if (typeof exports === 'object' && typeof module !== 'undefined') {
    module.exports = factory(require('react'));
  } else if (typeof define === 'function' && define.amd) {
    define(['react'], factory);
  } else {
    const React = global.React || (typeof window !== 'undefined' ? window.React : null);
    const mod = factory(React);
    global.LatticeLoader = mod.LatticeLoader || mod;
    global.createLatticeLoader = mod.createLatticeLoader;
    if (typeof window !== 'undefined') {
      window.LatticeLoader = mod.LatticeLoader || mod;
      window.createLatticeLoader = mod.createLatticeLoader;
    }
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : this, function (React) {
  'use strict';

  const PATTERNS = {
    arrow: { 3: { cells: [1, 2, 3, 0, 1, 2, 1, 2, 3], loop: 7.2, scale: 1 } },
    dots: { 3: { cells: [0, 1, 2, 0, 1, 2, 0, 1, 2], loop: 3, scale: 2.4 } },
    ripple: { 3: { cells: [2, 1, 2, 1, 0, 1, 2, 1, 2], loop: 4.8, scale: 1.5 } },
    spiral: { 3: { cells: [0, 1, 2, 7, 8, 3, 6, 5, 4], loop: 9, scale: 1.2, lit: 0.35 } },
    orbit: {
      3: { cells: [0, 1, 2, 7, null, 3, 6, 5, 4], loop: 8, scale: 1.2 },
      4: { cells: [0, 1, 2, 3, 11, null, null, 4, 10, null, null, 5, 9, 8, 7, 6], loop: 6, scale: 1.2, lit: 0.45 }
    },
    snake: {
      3: { cells: [0, 1, 2, 5, 4, 3, 6, 7, 8], loop: 9, scale: 1, lit: 0.35 },
      4: { cells: [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12], loop: 16, scale: 1, lit: 0.25 }
    },
    sweep: { 4: { cells: [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6], loop: 5, scale: 1, lit: 0.45 } },
    spin: { 4: { cells: [0, 0, 1, 1, 0, 0, 1, 1, 3, 3, 2, 2, 3, 3, 2, 2], loop: 4, scale: 1.6, lit: 0.35 } },
    rain: { 4: { cells: [0, 2, 1, 3, 1, 3, 2, 4, 2, 4, 3, 5, 3, 5, 4, 6], loop: 4, scale: 1.2, lit: 0.35 } },
    pulse: { 4: { cells: [2, 1, 1, 2, 1, 0, 0, 1, 1, 0, 0, 1, 2, 1, 1, 2], loop: 2.4, scale: 2.5, lit: 0.45 } }
  };

  const DEFAULT_PATTERN = { 3: 'orbit', 4: 'sweep' };
  const MARKS = {
    3: { done: [2, 3, 5, 7], error: [0, 2, 4, 6, 8] },
    4: { done: [7, 8, 10, 13], error: [0, 3, 5, 6, 9, 10, 12, 15] }
  };

  const resolvePattern = (pattern, grid) => {
    if (typeof pattern === 'string') {
      const named = PATTERNS[pattern];
      return (named && named[grid]) || PATTERNS[DEFAULT_PATTERN[grid]][grid];
    }
    const cells = Array.from({ length: grid * grid }, (_, i) => (pattern && pattern.cells ? pattern.cells[i] : null));
    const max = Math.max(0, ...cells.filter(v => v != null));
    return {
      cells,
      loop: (pattern && pattern.loop) ?? max + 4.2,
      scale: (pattern && pattern.scale) ?? 1,
      lit: (pattern && pattern.lit) ?? 0.62
    };
  };

  const fmt = ds => (ds < 600 ? `${(ds / 10).toFixed(1)}s` : `${Math.floor(ds / 600)}m ${((ds % 600) / 10).toFixed(1)}s`);
  const spoken = ds =>
    ds < 600
      ? `${(ds / 10).toFixed(1)} seconds`
      : `${Math.floor(ds / 600)} minutes ${((ds % 600) / 10).toFixed(1)} seconds`;

  // ---------------------------------------------------------------------------
  // React Component Implementation
  // ---------------------------------------------------------------------------
  function LatticeLoader(props) {
    if (!React) {
      console.warn("React is not loaded, use createLatticeLoader for vanilla environments.");
      return null;
    }

    const {
      label = 'Thinking',
      doneLabel = 'Done in',
      errorLabel = 'Failed after',
      status = 'working',
      pattern = 'orbit',
      grid = 3,
      shape = 'round',
      color = 'currentColor',
      doneColor = '#22c55e',
      errorColor = '#ef4444',
      cellSize = 6,
      gap = 2,
      fontSize = 14,
      step = 90,
      idleOpacity = 0.15,
      glow = false,
      glowColor = '',
      showTimer = true,
      elapsed,
      className = '',
      style
    } = props || {};

    const n = grid === 4 ? 4 : 3;
    const pat = resolvePattern(pattern, n);
    const marks = MARKS[n];
    const d = step * pat.scale;
    const cycle = Math.round(pat.loop * d);

    const timerRef = React.useRef(null);
    const dsRef = React.useRef(0);
    const markRef = React.useRef('done');
    const mark = status === 'working' ? markRef.current : status;
    markRef.current = mark;
    const [announce, setAnnounce] = React.useState(`${label}, in progress`);

    const paint = ds => {
      dsRef.current = ds;
      if (timerRef.current) timerRef.current.textContent = fmt(ds);
    };

    React.useLayoutEffect(() => {
      if (elapsed != null) {
        paint(Math.round(elapsed * 10));
        return undefined;
      }
      if (status !== 'working') return undefined;
      const startedAt = performance.now();
      paint(0);
      const id = setInterval(() => paint(Math.floor((performance.now() - startedAt) / 100)), 100);
      return () => clearInterval(id);
    }, [status, elapsed]);

    React.useEffect(() => {
      if (status === 'working') setAnnounce(`${label}, in progress`);
      else setAnnounce(`${status === 'done' ? doneLabel : errorLabel}${showTimer ? ` ${spoken(dsRef.current)}` : ''}`);
    }, [status, label, doneLabel, errorLabel, showTimer]);

    const inlineStyle = Object.assign(
      {
        '--ll-n': n,
        '--ll-cell': `${cellSize}px`,
        '--ll-gap': `${gap}px`,
        '--ll-font': `${fontSize}px`,
        '--ll-color': color,
        '--ll-mark': status === 'error' ? errorColor : doneColor,
        '--ll-idle': idleOpacity,
        '--ll-glow': glowColor || color,
        '--ll-mark-glow': glowColor || (status === 'error' ? errorColor : doneColor),
        '--ll-cycle': `${cycle}ms`
      },
      style
    );

    return React.createElement(
      'span',
      {
        role: 'status',
        className: `lattice-loader${className ? ` ${className}` : ''}`,
        'data-status': status,
        'data-shape': shape,
        'data-glow': glow ? '' : undefined,
        style: inlineStyle
      },
      React.createElement(
        'span',
        { className: 'lattice-loader__grid', 'aria-hidden': 'true' },
        React.createElement(
          'span',
          { className: 'lattice-loader__layer lattice-loader__run' },
          pat.cells.map((unit, i) =>
            React.createElement('span', {
              key: i,
              className: 'lattice-loader__cell',
              'data-hole': unit == null ? '' : undefined,
              'data-lit': pat.lit && pat.lit !== 0.62 ? Math.round(pat.lit * 100) : undefined,
              style: unit == null ? undefined : { animationDelay: `${Math.round(unit * d)}ms` }
            })
          )
        ),
        React.createElement(
          'span',
          { className: 'lattice-loader__layer lattice-loader__mark' },
          pat.cells.map((_, i) =>
            React.createElement('span', {
              key: i,
              className: 'lattice-loader__cell',
              'data-on': marks[mark] && marks[mark].includes(i) ? '' : undefined
            })
          )
        )
      ),
      React.createElement(
        'span',
        { className: 'lattice-loader__label', 'aria-hidden': 'true' },
        React.createElement(
          'span',
          { className: 'lattice-loader__text', 'data-active': status === 'working' ? '' : undefined },
          label
        ),
        React.createElement(
          'span',
          { className: 'lattice-loader__text', 'data-active': status === 'done' ? '' : undefined },
          doneLabel
        ),
        React.createElement(
          'span',
          { className: 'lattice-loader__text', 'data-active': status === 'error' ? '' : undefined },
          errorLabel
        )
      ),
      showTimer
        ? React.createElement(
            'span',
            { ref: timerRef, className: 'lattice-loader__timer', 'aria-hidden': 'true' },
            '0.0s'
          )
        : null,
      React.createElement('span', { className: 'lattice-loader__sr' }, announce)
    );
  }

  // ---------------------------------------------------------------------------
  // Vanilla DOM Factory: createLatticeLoader
  // Allows direct integration into vanilla JS/HTML applications
  // ---------------------------------------------------------------------------
  function createLatticeLoader(container, initialProps) {
    if (!container) return null;

    let props = Object.assign(
      {
        label: 'Thinking',
        doneLabel: 'Done in',
        errorLabel: 'Failed after',
        status: 'working',
        pattern: 'orbit',
        grid: 3,
        shape: 'round',
        color: 'currentColor',
        doneColor: '#22c55e',
        errorColor: '#ef4444',
        cellSize: 6,
        gap: 2,
        fontSize: 14,
        step: 90,
        idleOpacity: 0.15,
        glow: false,
        glowColor: '',
        showTimer: true,
        className: ''
      },
      initialProps || {}
    );

    const rootEl = document.createElement('span');
    container.appendChild(rootEl);

    let intervalId = null;
    let ds = 0;
    let timerEl = null;

    function render() {
      const n = props.grid === 4 ? 4 : 3;
      const pat = resolvePattern(props.pattern, n);
      const marks = MARKS[n];
      const d = props.step * pat.scale;
      const cycle = Math.round(pat.loop * d);
      const mark = props.status === 'working' ? 'done' : props.status;

      rootEl.className = `lattice-loader${props.className ? ` ${props.className}` : ''}`;
      rootEl.setAttribute('role', 'status');
      rootEl.setAttribute('data-status', props.status);
      rootEl.setAttribute('data-shape', props.shape);
      if (props.glow) {
        rootEl.setAttribute('data-glow', '');
      } else {
        rootEl.removeAttribute('data-glow');
      }

      rootEl.style.setProperty('--ll-n', n);
      rootEl.style.setProperty('--ll-cell', `${props.cellSize}px`);
      rootEl.style.setProperty('--ll-gap', `${props.gap}px`);
      rootEl.style.setProperty('--ll-font', `${props.fontSize}px`);
      rootEl.style.setProperty('--ll-color', props.color);
      rootEl.style.setProperty('--ll-mark', props.status === 'error' ? props.errorColor : props.doneColor);
      rootEl.style.setProperty('--ll-idle', props.idleOpacity);
      rootEl.style.setProperty('--ll-glow', props.glowColor || props.color);
      rootEl.style.setProperty('--ll-mark-glow', props.glowColor || (props.status === 'error' ? props.errorColor : props.doneColor));
      rootEl.style.setProperty('--ll-cycle', `${cycle}ms`);

      let runCells = '';
      pat.cells.forEach(unit => {
        const holeAttr = unit == null ? ' data-hole' : '';
        const litAttr = pat.lit && pat.lit !== 0.62 ? ` data-lit="${Math.round(pat.lit * 100)}"` : '';
        const delayStyle = unit != null ? ` style="animation-delay: ${Math.round(unit * d)}ms;"` : '';
        runCells += `<span class="lattice-loader__cell"${holeAttr}${litAttr}${delayStyle}></span>`;
      });

      let markCells = '';
      const activeMarkIndices = marks[mark] || [];
      pat.cells.forEach((_, i) => {
        const onAttr = activeMarkIndices.includes(i) ? ' data-on' : '';
        markCells += `<span class="lattice-loader__cell"${onAttr}></span>`;
      });

      let html = `
        <span class="lattice-loader__grid" aria-hidden="true">
          <span class="lattice-loader__layer lattice-loader__run">${runCells}</span>
          <span class="lattice-loader__layer lattice-loader__mark">${markCells}</span>
        </span>
        <span class="lattice-loader__label" aria-hidden="true">
          <span class="lattice-loader__text"${props.status === 'working' ? ' data-active' : ''}>${escapeHtml(props.label)}</span>
          <span class="lattice-loader__text"${props.status === 'done' ? ' data-active' : ''}>${escapeHtml(props.doneLabel)}</span>
          <span class="lattice-loader__text"${props.status === 'error' ? ' data-active' : ''}>${escapeHtml(props.errorLabel)}</span>
        </span>
      `;

      if (props.showTimer) {
        html += `<span class="lattice-loader__timer" aria-hidden="true">${fmt(ds)}</span>`;
      }

      html += `<span class="lattice-loader__sr">${props.status === 'working' ? `${props.label}, in progress` : `${props.status === 'done' ? props.doneLabel : props.errorLabel} ${spoken(ds)}`}</span>`;

      rootEl.innerHTML = html;
      timerEl = rootEl.querySelector('.lattice-loader__timer');
    }

    function escapeHtml(s) {
      if (!s) return '';
      return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function startTimer() {
      stopTimer();
      ds = 0;
      const startedAt = performance.now();
      intervalId = setInterval(() => {
        ds = Math.floor((performance.now() - startedAt) / 100);
        if (timerEl) timerEl.textContent = fmt(ds);
      }, 100);
    }

    function stopTimer() {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    }

    render();
    if (props.status === 'working') {
      startTimer();
    }

    return {
      element: rootEl,
      setStatus(newStatus, newLabel) {
        props.status = newStatus;
        if (newLabel) {
          if (newStatus === 'working') props.label = newLabel;
          else if (newStatus === 'done') props.doneLabel = newLabel;
          else if (newStatus === 'error') props.errorLabel = newLabel;
        }
        if (newStatus === 'working') {
          startTimer();
        } else {
          stopTimer();
        }
        render();
      },
      update(newProps) {
        const wasWorking = props.status === 'working';
        props = Object.assign(props, newProps);
        if (props.status === 'working' && !wasWorking) {
          startTimer();
        } else if (props.status !== 'working') {
          stopTimer();
        }
        render();
      },
      destroy() {
        stopTimer();
        if (rootEl.parentNode) {
          rootEl.parentNode.removeChild(rootEl);
        }
      }
    };
  }

  return {
    LatticeLoader: LatticeLoader,
    createLatticeLoader: createLatticeLoader
  };
});
