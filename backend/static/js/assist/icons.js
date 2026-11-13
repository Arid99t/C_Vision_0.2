// Small inline SVG icons (no icon font, no network). Every icon is drawn on a
// 24x24 grid and inherits the text colour, so it works on any panel.

const svg = (body, extra = '') =>
  `<svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="2" `
  + `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" ${extra}>${body}</svg>`;

export const LEVEL_ICON = {
  danger: svg('<path d="M8.2 2.5h7.6l5.7 5.7v7.6l-5.7 5.7H8.2l-5.7-5.7V8.2z" fill="currentColor" stroke="none"/>'
    + '<path d="M8 8l8 8M16 8l-8 8" stroke="#1a0606" stroke-width="2.4"/>'),
  caution: svg('<path d="M12 3L22 20H2z" fill="currentColor" stroke="none"/>'
    + '<path d="M12 9.5v5" stroke="#201400" stroke-width="2.4"/><circle cx="12" cy="17.3" r="1.3" fill="#201400" stroke="none"/>'),
  info: svg('<circle cx="12" cy="12" r="10" fill="currentColor" stroke="none"/>'
    + '<path d="M12 11v6" stroke="#04121f" stroke-width="2.4"/><circle cx="12" cy="7.4" r="1.4" fill="#04121f" stroke="none"/>'),
  none: svg('<circle cx="12" cy="12" r="10" fill="currentColor" stroke="none"/>'
    + '<path d="M7.5 12.5l3.2 3.2 6-6.7" stroke="#03170c" stroke-width="2.4"/>'),
};

// GHS-style hazard marks: a red diamond with a simple black glyph on white
const diamond = (glyph) =>
  `<svg viewBox="0 0 24 24" width="1em" height="1em" aria-hidden="true">`
  + `<path d="M12 1.2L22.8 12 12 22.8 1.2 12z" fill="#fff" stroke="#e02424" stroke-width="2"/>`
  + `<g fill="#111" stroke="#111" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">${glyph}</g></svg>`;

export const HAZARD = {
  flammable: { label: 'Flammable', icon: diamond('<path d="M12 6c2 2.6 4 4.2 4 7.2a4 4 0 0 1-8 0c0-1.6.8-2.6 1.8-3.6.2 1.4.8 2 1.4 2.2C10.8 9.800 11 7.800 12 6z" stroke="none"/>') },
  oxidiser: { label: 'Oxidiser', icon: diamond('<circle cx="12" cy="15" r="3" fill="none"/><path d="M12 5.500c1.2 1.6 2.300 2.600 2.300 4.300a2.300 2.300 0 0 1-4.600 0c0-1.700 1.100-2.700 2.300-4.300z" stroke="none"/>') },
  corrosive: { label: 'Corrosive', icon: diamond('<path d="M7 8h4M13 8h4M9 8v2.500M15 8v2.500" fill="none"/><path d="M6.500 14h5v1.800h-5zM13 13.200l3.600.900-.500 2-3.600-.900z" stroke="none"/><path d="M6 17.800h12" fill="none"/>') },
  toxic: { label: 'Toxic', icon: diamond('<path d="M12 6a4 4 0 0 0-2.400 7.200V15h4.800v-1.800A4 4 0 0 0 12 6z" stroke="none"/><circle cx="10.500" cy="10" r="1" fill="#fff" stroke="none"/><circle cx="13.500" cy="10" r="1" fill="#fff" stroke="none"/><path d="M8 16.500l8 2M16 16.500l-8 2" fill="none"/>') },
  irritant: { label: 'Irritant / harmful', icon: diamond('<path d="M12 6.500v6.500" stroke-width="2.600" fill="none"/><circle cx="12" cy="16.600" r="1.500" stroke="none"/>') },
  health: { label: 'Serious health hazard', icon: diamond('<circle cx="12" cy="7.800" r="1.900" stroke="none"/><path d="M8.500 18l1-6.500h5l1 6.500z" stroke="none"/><path d="M12 11.500l-1.200 2 2.400.800L12 16.500" stroke="#fff" stroke-width="1" fill="none"/>') },
  environment: { label: 'Harmful to the environment', icon: diamond('<path d="M6.500 16.500h11" fill="none"/><path d="M9 16V9.500M9 9.500l-2 1.500M9 11l2-1.500M9 12.500l-2 1.500" fill="none" stroke-width="1.100"/><path d="M12.500 15c1.200-1.400 3-1.400 4.200 0-1.200 1.200-3 1.200-4.200 0z" stroke="none"/>') },
};

export const PPE = {
  gloves: { label: 'Gloves', icon: svg('<path d="M8 21v-5l-2-3V8a1.300 1.300 0 0 1 2.600 0V5a1.300 1.300 0 0 1 2.600 0V4a1.300 1.300 0 0 1 2.600 0v1.500a1.300 1.300 0 0 1 2.600 0V14l-1.400 3v4z"/>') },
  goggles: { label: 'Goggles', icon: svg('<rect x="2.500" y="8" width="8" height="7" rx="3"/><rect x="13.500" y="8" width="8" height="7" rx="3"/><path d="M10.500 11h3"/>') },
  'lab coat': { label: 'Lab coat', icon: svg('<path d="M8 3l4 3 4-3 4 3v15H4V6z"/><path d="M12 6v15"/>') },
};

export const KIND = {
  bottle: svg('<path d="M10 2h4v4l2 3v11a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2V9l2-3z"/><path d="M8 13h8"/>'),
  flask: svg('<path d="M9 2h6M10 2v6l-5 10a2 2 0 0 0 1.800 3h10.400a2 2 0 0 0 1.800-3L14 8V2"/><path d="M7.500 15h9"/>'),
  box: svg('<path d="M3 7.500L12 3l9 4.500v9L12 21l-9-4.500z"/><path d="M3 7.500l9 4.500 9-4.500M12 12v9"/>'),
  carton: svg('<path d="M7 8l2-5h6l2 5v12a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1z"/><path d="M7 8h10M9 3l3 5"/>'),
  can: svg('<rect x="6.500" y="3" width="11" height="18" rx="2.500"/><path d="M6.500 7h11M6.500 17h11"/>'),
  cup: svg('<path d="M5 7h12l-1.200 12a2 2 0 0 1-2 1.800H8.200a2 2 0 0 1-2-1.800z"/><path d="M17 9.500h1.500a2.500 2.500 0 0 1 0 5h-2"/><path d="M9 3.500c0 1 .800 1 .800 2M13 3.500c0 1 .800 1 .800 2"/>'),
  pill: svg('<rect x="2.500" y="8.500" width="19" height="7" rx="3.500" transform="rotate(-35 12 12)"/><path d="M9.500 8.400l5 7.200"/>'),
  drop: svg('<path d="M12 3c3 4 5.500 6.800 5.500 10.200a5.500 5.500 0 0 1-11 0C6.500 9.800 9 7 12 3z"/>'),
  unknown: svg('<circle cx="12" cy="12" r="9"/><path d="M9.500 9.500a2.500 2.500 0 1 1 3.500 2.300c-.700.400-1 .900-1 1.700"/><circle cx="12" cy="16.800" r=".6" fill="currentColor"/>'),
};

export const UI = {
  safety: svg('<path d="M12 2.500l8 3v6c0 5-3.400 8.600-8 10-4.600-1.400-8-5-8-10v-6z"/><path d="M8.500 12l2.500 2.500 4.500-5"/>'),
  electronics: svg('<rect x="7" y="7" width="10" height="10" rx="1.500"/><path d="M10 7V3.500M14 7V3.500M10 20.500V17M14 20.500V17M7 10H3.500M7 14H3.500M20.500 10H17M20.500 14H17"/>'),
  access: svg('<circle cx="12" cy="4.500" r="2"/><path d="M4.500 8.500l7.500 1.500 7.500-1.500M12 10v5M8.500 21.500L12 15l3.500 6.500"/>'),
  captions: svg('<rect x="2.500" y="5" width="19" height="14" rx="3"/><path d="M10.500 10.500a2.200 2.200 0 1 0 0 3M17 10.500a2.200 2.200 0 1 0 0 3"/>'),
  read: svg('<path d="M3 5.500c3-1.500 6-1.500 9 0v14c-3-1.500-6-1.500-9 0zM21 5.500c-3-1.500-6-1.500-9 0v14c3-1.500 6-1.500 9 0z"/>'),
  speak: svg('<path d="M4 9.500v5h3.500l4.500 4v-13l-4.500 4z"/><path d="M15.500 9a4 4 0 0 1 0 6M18 6.500a7.500 7.500 0 0 1 0 11"/>'),
  mic: svg('<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.500 11.500a6.500 6.500 0 0 0 13 0M12 18v3"/>'),
  resistor: svg('<path d="M2 12h4l1.500-4 3 8 3-8 3 8L18 12h4"/>'),
  chip: svg('<rect x="6" y="5" width="12" height="14" rx="1.500"/><path d="M6 8.500H3M6 12H3M6 15.500H3M21 8.500h-3M21 12h-3M21 15.500h-3"/><circle cx="9" cy="8" r=".8" fill="currentColor"/>'),
  close: svg('<path d="M6 6l12 12M18 6L6 18"/>'),
  clear: svg('<path d="M4 7h16M9 7V4.500h6V7M6.500 7l1 13h9l1-13"/>'),
  list: svg('<path d="M8 6.500h12M8 12h12M8 17.500h12"/><circle cx="4" cy="6.500" r=".9" fill="currentColor"/><circle cx="4" cy="12" r=".9" fill="currentColor"/><circle cx="4" cy="17.500" r=".9" fill="currentColor"/>'),
  pause: svg('<path d="M8 5v14M16 5v14"/>'),
  allergy: svg('<circle cx="12" cy="12" r="9"/><path d="M5.800 5.800l12.400 12.400"/>'),
  pen: svg('<path d="M4 20l1-4.500L16.500 4l3.500 3.500L8.500 19z"/><path d="M14 6.500l3.500 3.500"/>'),
  text: svg('<path d="M5 6.500V5h14v1.500M12 5v14M9 19h6"/>'),
  barcode: svg('<path d="M4 6v12M7 6v12M10.500 6v9M13.500 6v9M17 6v12M20 6v12M9.500 18h5"/>'),
};
