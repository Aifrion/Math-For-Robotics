/* CSE 276C notes — sidebar nav + iframe auto-sizing */
(function () {
  var PAGES = [
    { group: 'Course' },
    { href: 'index.html', title: 'Home' },
    { group: 'Lectures' },
    { href: 'lec2.html', title: '2 · Linear Systems', soon: true },
    { href: 'lec3.html', title: '3 · Interpolation', soon: true },
    { href: 'lec4.html', title: '4 · Function Approximation', soon: true },
    { href: 'lec5.html', title: '5 · Root Finding' },
    { href: 'lec6.html', title: '6 · Optimization' },
    { href: 'lec7.html', title: '7 · Integration & ODEs' },
    { href: 'lec8.html', title: '8 · Subspace Methods' },
    { group: 'Homework' },
    { href: 'hw2.html', title: 'HW2 · Trajectory Interpolation' }
  ];

  var here = location.pathname.split('/').pop() || 'index.html';
  var nav = document.createElement('nav');
  nav.className = 'side';
  var h = '<div class="brand">Math for Robotics</div>' +
          '<div class="sub">CSE 276C · interactive notes</div>';
  PAGES.forEach(function (p) {
    if (p.group) { h += '<div class="group">' + p.group + '</div>'; return; }
    if (p.soon) { h += '<span class="soon">' + p.title + '</span>'; return; }
    h += '<a href="' + p.href + '"' + (p.href === here ? ' class="on"' : '') + '>' + p.title + '</a>';
  });
  nav.innerHTML = h;

  var layout = document.querySelector('.layout');
  if (layout) layout.insertBefore(nav, layout.firstChild);

  var btn = document.createElement('button');
  btn.id = 'menuBtn';
  btn.textContent = 'Menu';
  btn.onclick = function () { nav.classList.toggle('open'); };
  document.body.appendChild(btn);
  nav.addEventListener('click', function (e) {
    if (e.target.tagName === 'A') nav.classList.remove('open');
  });

  /* iframe auto-size: demos post {demoHeight: N} */
  window.addEventListener('message', function (e) {
    if (!e.data || typeof e.data.demoHeight !== 'number') return;
    var frames = document.querySelectorAll('iframe.demo');
    for (var i = 0; i < frames.length; i++) {
      if (frames[i].contentWindow === e.source) {
        frames[i].style.height = (e.data.demoHeight + 2) + 'px';
      }
    }
  });

  /* KaTeX auto-render once it loads */
  window.addEventListener('DOMContentLoaded', function () {
    if (window.renderMathInElement) {
      renderMathInElement(document.body, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\(', right: '\\)', display: false }
        ]
      });
    }
  });
})();
