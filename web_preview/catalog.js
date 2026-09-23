"use strict";
// Catalog presentation is separate from main's matching client.
(() => {
const $ = id => document.getElementById(id);
const form = $('request-form');
const money = new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 2});
let profiles = [];
let visibleProfiles = 0;
let selectedProfile = null;
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function chooseCategory(category, city) {
  if ($('request-fields').disabled) return;
  $('category').value = category;
  if (city) $('city').value = city;
  form.dispatchEvent(new Event('input'));
  document.querySelector('#matching').scrollIntoView({behavior: 'auto', block: 'start'});
  $('category').focus({preventScroll: true});
  // Use main's existing submit handler, including cancellation of older requests.
  form.requestSubmit();
}

// Browse the original catalog, independently of the ranked recommendation cards.
const categoryArt = [
  ['Ведущий', 'Ведущие', 'M12 16v5m-4 0h8M8 5a4 4 0 0 1 8 0v6a4 4 0 0 1-8 0Zm-3 5v1a7 7 0 0 0 14 0v-1'],
  ['Фотограф', 'Фотографы', 'M3 7h4l2-3h6l2 3h4v13H3ZM16 13a4 4 0 1 1-8 0 4 4 0 0 1 8 0'],
  ['Банкетный зал', 'Банкетные залы', 'M3 21h18M5 21V8l7-5 7 5v13M9 21v-7h6v7M8 10h1m6 0h1'],
  ['Флорист', 'Флористы', 'M12 21v-8m0 5c-5 0-7-2-7-5 4 0 7 1 7 5Zm0-2c5 0 7-2 7-5-4 0-7 1-7 5ZM12 3c-5-5-8 4-3 6-2 5 8 5 6 0 5-2 2-11-3-6Z'],
  ['Видеограф', 'Видеографы', 'M3 6h12v13H3Zm12 4 6-3v11l-6-3Z'],
  ['Декоратор', 'Декораторы', 'M4 21V10a8 8 0 0 1 16 0v11M8 21V10a4 4 0 0 1 8 0v11M2 21h6m8 0h6'],
];

function initials(name) {
  return name.trim().split(/\s+/).slice(0, 2).map(part => part[0]).join('');
}

function renderCatalog() {
  const fragment = document.createDocumentFragment();
  const batch = profiles.slice(visibleProfiles, visibleProfiles + 6);
  for (const profile of batch) {
    const article = node('article', undefined, 'profile-card');
    const button = node('button', undefined, 'profile-open');
    button.type = 'button';
    button.setAttribute('aria-label', `Открыть профиль: ${profile.anon_name}`);
    const art = node('div', undefined, 'profile-art');
    const monogram = node('span', initials(profile.anon_name), 'profile-initials');
    monogram.setAttribute('aria-hidden', 'true');
    art.append(monogram, node('span', profile.city, 'profile-city'));
    if (profile.synthetic) art.append(node('span', 'Синтетический профиль', 'synthetic-label'));
    const info = node('div', undefined, 'profile-info');
    const bottom = node('div', undefined, 'profile-bottom');
    bottom.append(node('span', `от ${money.format(profile.price_from_kzt)} ₸`, 'profile-price'), node('span', 'Профиль ↗', 'profile-more'));
    info.append(node('p', profile.categories.join(' · '), 'profile-category'), node('h3', profile.anon_name, 'profile-name'), node('p', profile.description, 'profile-excerpt'), bottom);
    button.append(art, info);
    button.addEventListener('click', () => openProfile(profile));
    article.append(button);
    fragment.append(article);
  }
  $('profile-grid').append(fragment);
  visibleProfiles += batch.length;
  $('more-profiles').hidden = visibleProfiles >= profiles.length;
  $('more-profiles').textContent = `Показать ещё · ${visibleProfiles} из ${profiles.length} ↓`;
}

function openProfile(profile) {
  selectedProfile = profile;
  const header = node('div', undefined, 'dialog-profile-header');
  const title = node('div');
  const name = node('h2', profile.anon_name);
  name.id = 'profile-name';
  title.append(name, node('p', `${profile.city} · ${profile.categories.join(' · ')}`, 'dialog-categories'));
  header.append(node('div', initials(profile.anon_name), 'dialog-avatar'), title);
  const facts = node('dl', undefined, 'profile-facts');
  for (const [label, value] of [
    ['Стоимость за мероприятие', `от ${money.format(profile.price_from_kzt)} ₸`],
    ['Языки', profile.languages.join(', ')],
    ['Форматы', profile.event_formats.join(', ')],
    ['Длительность', profile.max_hours === null ? 'Без привязки к присутствию' : `До ${profile.max_hours} ч`],
  ]) {
    const pair = node('div');
    pair.append(node('dt', label), node('dd', value));
    facts.append(pair);
  }
  const flags = node('ul', undefined, 'profile-flags');
  for (const [flag, text] of [['synthetic', 'Синтетический профиль из демо-каталога.'], ['city_imputed', 'Город проставлен при подготовке датасета.'], ['price_imputed', 'Цена проставлена при подготовке датасета.']]) {
    if (profile[flag]) flags.append(node('li', text));
  }
  $('profile-content').replaceChildren(header, facts, node('p', profile.description, 'profile-bio'), flags);
  $('profile-dialog').showModal();
}
$('close-profile').addEventListener('click', () => $('profile-dialog').close());
$('match-profile').addEventListener('click', () => {
  $('profile-dialog').close();
  if (selectedProfile) chooseCategory(selectedProfile.categories[0], selectedProfile.city);
});
$('more-profiles').addEventListener('click', renderCatalog);

async function loadCatalog() {
  try {
    const response = await fetch('/api/catalog');
    if (!response.ok) throw new Error('Catalog unavailable');
    profiles = (await response.json()).profiles;
    $('profile-total').textContent = profiles.length;
    $('category-total').textContent = new Set(profiles.flatMap(p => p.categories)).size;
    for (const [category, label, path] of categoryArt) {
      const tile = node('button', undefined, 'category-tile');
      tile.type = 'button';
      tile.setAttribute('aria-label', `Подобрать: ${label}`);
      const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      icon.setAttribute('viewBox', '0 0 24 24');
      icon.setAttribute('class', 'category-icon');
      icon.setAttribute('aria-hidden', 'true');
      const shape = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      shape.setAttribute('d', path);
      icon.append(shape);
      const count = profiles.reduce((sum, p) => sum + Number(p.categories.includes(category)), 0);
      tile.append(icon, node('strong', label), node('small', `В каталоге: ${count}`), node('span', '↗', 'category-arrow'));
      tile.addEventListener('click', () => chooseCategory(category));
      $('category-grid').append(tile);
    }
    renderCatalog();
  } catch {
    $('catalog-error').hidden = false;
    $('catalog-error').textContent = 'Не удалось открыть каталог. Обновите страницу или воспользуйтесь формой подбора ниже.';
  }
}

loadCatalog();
})();
