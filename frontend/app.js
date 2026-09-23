"use strict";

const PRESETS = [
  { label: "Ведущий · плотная категория", icon: "✦", values: { city: "Алматы", date: "2026-09-23", event_type: "свадьба", category: "Ведущий", budget: 2000000, preferences: "спокойный стиль, европейская подача и уважение к традициям" } },
  { label: "Флорист · редкая категория", icon: "❀", values: { city: "Алматы", date: "2026-10-04", event_type: "свадьба", category: "Флорист", budget: 1000000, preferences: "авторское цветочное оформление" } },
  { label: "Никто не прошёл условия", icon: "○", values: { city: "Алматы", date: "2026-10-01", event_type: "свадьба", category: "Ведущий", budget: 1, preferences: "" } },
  { label: "Нет категории в городе", icon: "⌕", values: { city: "Зарубежье", date: "2026-10-04", event_type: "свадьба", category: "Флорист", budget: 1000000, preferences: "" } },
  { label: "Сравнить · 1 октября", icon: "◷", values: { city: "Алматы", date: "2026-10-01", event_type: "свадьба", category: "Ведущий", budget: 1000000, preferences: "" } },
  { label: "Сравнить · 2 октября", icon: "◷", values: { city: "Алматы", date: "2026-10-02", event_type: "свадьба", category: "Ведущий", budget: 1000000, preferences: "" } },
];

const $ = (id) => document.getElementById(id);
const form = $("request-form");
const resultRoot = $("result-content");
const presetRoot = $("preset-list");
const money = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 });
let currentRequest = 0;
let controller = null;

function node(tag, className, value) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (value !== undefined && value !== null) element.textContent = String(value);
  return element;
}

function append(parent, ...children) {
  parent.append(...children);
  return parent;
}

function fillSelect(id, values, initial) {
  const select = $(id);
  select.replaceChildren();
  for (const value of values) {
    const option = node("option", "", value);
    option.value = value;
    select.append(option);
  }
  if (values.includes(initial)) select.value = initial;
}

function formatDate(value) {
  const parts = String(value || "").split("-");
  return parts.length === 3 ? `${parts[2]}.${parts[1]}.${parts[0]}` : String(value || "");
}

function applyValues(values) {
  $("city").value = values.city;
  $("date").value = values.date;
  $("event-type").value = values.event_type;
  $("category").value = values.category;
  $("budget").value = values.budget;
  $("language").value = "";
  $("duration").value = "";
  $("preferences").value = values.preferences;
}

function setActivePreset(index) {
  presetRoot.querySelectorAll("button").forEach((button, current) => {
    button.classList.toggle("active", current === index);
    button.setAttribute("aria-pressed", current === index ? "true" : "false");
  });
}

function createPresets() {
  PRESETS.forEach((preset, index) => {
    const button = node("button", "preset");
    button.type = "button";
    button.disabled = true;
    button.setAttribute("aria-pressed", "false");
    append(button, node("span", "preset-icon", preset.icon), node("span", "", preset.label));
    button.addEventListener("click", () => {
      applyValues(preset.values);
      setActivePreset(index);
      submitRequest();
    });
    presetRoot.append(button);
  });
}

function readRequest() {
  return {
    city: $("city").value,
    date: $("date").value,
    event_type: $("event-type").value,
    category: $("category").value,
    budget: Number($("budget").value),
    language: $("language").value || null,
    duration_hours: $("duration").value ? Number($("duration").value) : null,
    preferences: $("preferences").value.trim(),
  };
}

function showLoading() {
  $("ai-availability").textContent = "Подбор выполняется…";
  const box = node("div", "loading-state");
  const spinner = node("div", "spinner");
  spinner.setAttribute("aria-hidden", "true");
  append(box, spinner, node("h3", "", "Проверяем подрядчиков"),
    node("p", "", "Сверяем календарь, бюджет и формат события, затем готовим объяснения."));
  resultRoot.replaceChildren(box);
}

function showError(message) {
  $("ai-availability").textContent = "Последний подбор: ошибка";
  const box = node("div", "error-state");
  box.setAttribute("role", "alert");
  const icon = node("div", "initial-icon", "!");
  icon.setAttribute("aria-hidden", "true");
  append(box, icon, node("h3", "", "Не удалось получить результат"),
    node("p", "", message || "Попробуйте повторить запрос."));
  resultRoot.replaceChildren(box);
}

function addPill(parent, value) {
  parent.append(node("span", "tiny-pill", value));
}

function outcomeHeading(status) {
  if (status === "matched") return "Подрядчики найдены";
  if (status === "no_category_in_city") return "Категории в городе нет";
  return "Никто не прошёл условия";
}

function renderOutcome(result) {
  const status = result.status;
  const box = node("div", `outcome ${status === "no_matches" ? "empty" : status === "no_category_in_city" ? "no-category" : ""}`);
  const top = node("div", "outcome-top");
  const symbol = status === "matched" ? "✓" : status === "no_category_in_city" ? "⌕" : "!";
  append(top, node("span", "outcome-icon", symbol), node("h3", "", outcomeHeading(status)));
  append(box, top, node("p", "", result.message || ""));
  const meta = node("div", "outcome-meta");
  if (status === "matched") addPill(meta, `${Math.min((result.cards || []).length, 3)} из 3 рекомендаций`);
  if (result.counts && typeof result.counts.city_category === "number") {
    addPill(meta, `${result.counts.city_category} в городе и категории`);
  }
  if (result.request && result.request.date) addPill(meta, formatDate(result.request.date));
  if (meta.children.length) box.append(meta);
  return box;
}

function factLabel(fact) {
  const labels = {
    city: "Город", categories: "Категории", busy_dates: "Доступность",
    price_from_kzt: "Цена от", event_formats: "Форматы", languages: "Языки",
    max_hours: "Длительность", description: "Описание",
  };
  return labels[fact.field] || "Факт";
}

function factText(fact) {
  if (fact.field === "busy_dates") return `Свободен на ${formatDate(fact.value)}`;
  if (fact.field === "price_from_kzt") return `${money.format(fact.value)} ₸`;
  if (fact.field === "max_hours") return fact.value == null ? "Работа не привязана к часам присутствия" : `До ${fact.value} ч`;
  if (Array.isArray(fact.value)) return fact.value.join(", ");
  return String(fact.value ?? "—");
}

function renderDetails(card) {
  const details = node("details", "");
  details.append(node("summary", "", "Проверить балл и факты"));
  const content = node("div", "detail-content");
  const score = Number(card.score);
  if (Number.isFinite(score)) {
    content.append(node("p", "", `Итоговый балл: ${score.toFixed(2)}`));
  }
  if (card.score_details) {
    const part = card.score_details;
    const semantic = Number(part.semantic_points);
    const budget = Number(part.budget_points);
    if (Number.isFinite(semantic) && Number.isFinite(budget)) {
      content.append(node("p", "", `Сходство описания: ${semantic.toFixed(2)} · запас бюджета: ${budget.toFixed(2)}`));
    }
  }
  if (Array.isArray(card.evidence) && card.evidence.length) {
    const list = node("ul", "");
    for (const fact of card.evidence) list.append(node("li", "", `${factLabel(fact)}: ${factText(fact)}`));
    content.append(list);
  }
  if (Array.isArray(card.warnings)) {
    for (const warning of card.warnings) content.append(node("p", "", warning));
  }
  details.append(content);
  return details;
}

function renderCard(card, index) {
  const article = node("article", "result-card");
  const top = node("div", "card-topline");
  const identity = node("div", "");
  append(identity, node("span", "card-number", `РЕКОМЕНДАЦИЯ ${String(index + 1).padStart(2, "0")}`),
    node("h3", "card-name", card.name || "Подрядчик"),
    node("p", "card-location", `${card.category || ""} · ${card.city || ""}`));
  const price = node("div", "card-price");
  append(price, node("small", "", "Цена от"), node("span", "", `${money.format(Number(card.price_from_kzt) || 0)} ₸`));
  append(top, identity, price);
  append(article, top, node("p", "card-explanation", card.explanation || ""));
  const flags = node("div", "card-flags");
  if (card.synthetic) flags.append(node("span", "flag synthetic", "Синтетический профиль"));
  if (card.city_imputed) flags.append(node("span", "flag", "Город добавлен в данных"));
  if (card.price_imputed) flags.append(node("span", "flag", "Цена добавлена в данных"));
  if (flags.children.length) article.append(flags);
  article.append(renderDetails(card));
  return article;
}

function renderExclusions(excluded) {
  if (!Array.isArray(excluded) || !excluded.length) return null;
  const details = node("details", "exclusions");
  details.append(node("summary", "", `Почему исключены другие · ${excluded.length}`));
  const list = node("ul", "excluded-list");
  for (const item of excluded) {
    const row = node("li", "");
    row.append(node("strong", "", item.anon_name || "Подрядчик"));
    const reasons = Array.isArray(item.reasons) ? item.reasons.map((reason) =>
      typeof reason === "string" ? reason : reason.message).filter(Boolean) : [];
    row.append(node("span", "", reasons.join("; ")));
    list.append(row);
  }
  details.append(list);
  return details;
}

function renderFirstPlaceReason(result) {
  const [first, second] = result.cards || [];
  if (!first) return null;
  const box = node("section", "first-place-reason");
  const heading = node("div", "first-place-heading");
  append(heading, node("span", "first-place-icon", "1"),
    node("h3", "", second ? "Почему этот подрядчик первый" : "Почему выбран этот подрядчик"));
  box.append(heading);

  const semanticLabel = result.ai?.mode === "openai" ? "Смысловое совпадение" : "Совпадение по словам";
  const hasPreferences = Boolean(result.request?.preferences?.trim());
  const matchTarget = hasPreferences ? "пожеланиям" : "категории и формату события";
  const comparisonTarget = hasPreferences ? "с пожеланиями" : "с категорией и форматом события";
  const firstParts = first.score_details || {};
  if (second) {
    const secondParts = second.score_details || {};
    const lead = `${first.name}: ${Number(first.score).toFixed(2)} балла; ${second.name}: ${Number(second.score).toFixed(2)}.`;
    box.append(node("p", "first-place-lead", lead));
    const semanticDifference = Number(firstParts.semantic_points) - Number(secondParts.semantic_points);
    const budgetDifference = Number(firstParts.budget_points) - Number(secondParts.budget_points);
    let reason;
    if (Math.abs(Number(first.score) - Number(second.score)) < 0.000001) {
      reason = "Баллы равны; порядок определён ID профиля.";
    } else if (semanticDifference > 0 && budgetDifference > 0) {
      reason = `Первый получил больше баллов за соответствие описания ${matchTarget} и за запас бюджета.`;
    } else if (semanticDifference > 0) {
      reason = budgetDifference < 0
        ? `Более сильное совпадение описания ${comparisonTarget} перевесило меньший запас бюджета.`
        : `Первый выше за счёт совпадения описания ${comparisonTarget}.`;
    } else if (budgetDifference > 0) {
      reason = semanticDifference < 0
        ? `Больший запас бюджета перевесил меньший балл за совпадение описания ${comparisonTarget}.`
        : "Первый выше за счёт большего запаса бюджета.";
    } else {
      reason = "Порядок рассчитан по итоговому баллу из двух показателей.";
    }
    box.append(node("p", "first-place-reason-text", reason));
    box.append(node("p", "first-place-breakdown",
      `${semanticLabel}: ${Number(firstParts.semantic_points).toFixed(2)} против ${Number(secondParts.semantic_points).toFixed(2)} балла; запас бюджета: ${Number(firstParts.budget_points).toFixed(2)} против ${Number(secondParts.budget_points).toFixed(2)}.`));
    const weights = firstParts.weights || {};
    if (Number.isFinite(Number(weights.semantic)) && Number.isFinite(Number(weights.budget)) && result.request.budget > 0) {
      const textMeasure = result.ai?.mode === "openai" ? "сходство эмбеддингов" : "совпадение слов";
      box.append(node("p", "first-place-formula",
        `Расчёт: ${textMeasure} × ${weights.semantic} + (1 − цена «от» / бюджет) × ${weights.budget}.`));
    }
  } else {
    box.append(node("p", "first-place-lead", "Это единственный подрядчик, прошедший условия запроса."));
  }

  const facts = Array.isArray(first.evidence) ? first.evidence : [];
  const hasFact = (field) => facts.some((fact) => fact.field === field);
  const proof = node("div", "first-place-proof");
  proof.append(node("h4", "", "Что подтверждено данными каталога"));
  const list = node("ul", "");
  if (hasFact("price_from_kzt")) {
    list.append(node("li", "", `Цена от ${money.format(first.price_from_kzt)} ₸ при вашем бюджете ${money.format(result.request.budget)} ₸.`));
  }
  if (hasFact("event_formats")) {
    list.append(node("li", "", `Формат «${result.request.event_type}» указан в профиле.`));
  }
  if (hasFact("busy_dates")) {
    list.append(node("li", "", `${formatDate(result.request.date)} не отмечено как занятое в календаре CSV.`));
  }
  if (result.request.language && hasFact("languages")) {
    list.append(node("li", "", `Язык «${result.request.language}» указан в профиле.`));
  }
  if (result.request.duration_hours && hasFact("max_hours")) {
    const hours = facts.find((fact) => fact.field === "max_hours").value;
    list.append(node("li", "", hours == null
      ? "Работа не привязана к часам присутствия."
      : `Лимит подрядчика ${hours} ч при запросе ${result.request.duration_hours} ч.`));
  }
  if (first.synthetic) list.append(node("li", "", "Это синтетический профиль из исходного датасета."));
  if (first.city_imputed) list.append(node("li", "", "Город был добавлен при подготовке данных."));
  if (first.price_imputed) list.append(node("li", "", "Цена была добавлена при подготовке данных."));
  if (list.children.length) proof.append(list);
  const excerpt = facts.find((fact) => fact.field === "description")?.value;
  if (excerpt) {
    proof.append(node("p", "first-place-source", "Фрагмент исходного описания:"));
    proof.append(node("blockquote", "first-place-quote", `«${excerpt}»`));
  }
  if (proof.children.length > 1) box.append(proof);
  return box;
}

function renderResult(result) {
  const children = [renderOutcome(result)];
  const ai = result.ai || {};
  const mode = ai.mode === "openai" ? `Смысловое сравнение: OpenAI ${ai.model || ""}` :
    ai.mode === "lexical" ? "Сравнение текстов по словам" : "Ранжирование не потребовалось";
  $("ai-availability").textContent = `Последний подбор: ${ai.mode === "openai" ? "эмбеддинги OpenAI" :
    ai.mode === "lexical" ? "сравнение по словам" : "ранжирование не потребовалось"}`;
  children.push(node("p", "mode-note", ai.message ? `${mode}. ${ai.message}` : mode));
  if (result.status === "matched") {
    children.push(renderFirstPlaceReason(result));
    const cards = node("div", "cards");
    for (const [index, card] of (result.cards || []).slice(0, 3).entries()) cards.append(renderCard(card, index));
    children.push(cards);
  }
  const exclusions = renderExclusions(result.excluded);
  if (exclusions) children.push(exclusions);
  resultRoot.replaceChildren(...children);
}

async function submitRequest() {
  validateDuration();
  if (!form.reportValidity()) return;
  if (controller) controller.abort();
  controller = new AbortController();
  const thisRequest = ++currentRequest;
  $("submit-button").disabled = true;
  showLoading();
  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readRequest()),
      signal: controller.signal,
    });
    const result = await response.json();
    if (thisRequest !== currentRequest) return;
    if (!response.ok || result.error) throw new Error(result.error || `Ошибка сервера (${response.status})`);
    renderResult(result);
  } catch (error) {
    if (error.name !== "AbortError" && thisRequest === currentRequest) {
      showError(error instanceof TypeError ? "Связь с локальным сервером потеряна. Проверьте, что он запущен." : error.message);
    }
  } finally {
    if (thisRequest === currentRequest) $("submit-button").disabled = false;
  }
}

function validateDuration() {
  const duration = $("duration");
  const value = duration.valueAsNumber;
  duration.setCustomValidity(duration.value !== "" && (!Number.isFinite(value) || value <= 0)
    ? "Укажите длительность больше нуля или оставьте поле пустым." : "");
}

async function initialize() {
  createPresets();
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    setActivePreset(-1);
    submitRequest();
  });
  form.addEventListener("input", () => setActivePreset(-1));
  form.addEventListener("change", () => setActivePreset(-1));
  $("duration").addEventListener("input", validateDuration);
  try {
    const response = await fetch("/api/meta");
    const meta = await response.json();
    if (!response.ok || meta.error) throw new Error(meta.error || `Ошибка сервера (${response.status})`);
    fillSelect("city", meta.cities || [], "Алматы");
    fillSelect("event-type", meta.event_types || [], "свадьба");
    fillSelect("category", meta.categories || [], "Ведущий");
    fillSelect("language", ["", ...(meta.languages || [])], "");
    $("language").options[0].textContent = "Любой";
    const start = meta.calendar && meta.calendar.start;
    const end = meta.calendar && meta.calendar.end;
    $("date").min = start || "";
    $("date").max = end || "";
    $("date").value = start || "";
    $("budget").value = 2000000;
    $("catalog-summary").textContent = `${meta.profiles || 0} профилей · ${formatDate(start)}–${formatDate(end)}`;
    $("ai-availability").textContent = meta.ai_available
      ? "Ключ OpenAI задан · фактический режим появится после подбора"
      : "Ключ OpenAI не задан · доступен подбор по словам";
    $("request-fields").disabled = false;
    presetRoot.querySelectorAll("button").forEach((button) => { button.disabled = false; });
  } catch (error) {
    $("catalog-summary").textContent = "Каталог недоступен";
    showError(error instanceof TypeError ? "Не удалось соединиться с локальным сервером." : error.message);
  }
}

initialize();
