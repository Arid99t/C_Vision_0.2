"""The Assist app (/assist) in a real (headless) browser.

Covers the three modes end to end on drawn test pictures and a synthetic
speech clip, the safety rule engine, the two-item tray, and the touchpad /
keyboard navigation at the glasses' 640x360 screen size.

These prove the app works and stays working. They do not measure accuracy on
real packaging, real resistors or real voices: see README "Assist app".
"""
import os
import shutil

import pytest

from tests.webharness import ROOT, browser_page, playwright_available, skip_unavailable

if not playwright_available():
    skip_unavailable("playwright / uvicorn not installed")

from tests import assist_scenes as sc  # noqa: E402

MODELS = os.path.join(ROOT, "backend", "static", "models")
HAS_TEXT = all(os.path.exists(os.path.join(MODELS, "assist", f)) for f in ("ocr_det.onnx", "ocr_rec.onnx"))
HAS_SPEECH = all(os.path.exists(os.path.join(MODELS, "assist", f))
                 for f in ("asr_encoder.onnx", "asr_decoder.onnx", "asr_vocab.json"))
HAS_RESISTOR = os.path.exists(os.path.join(MODELS, "resistor.onnx"))
HAS_RUNTIME = os.path.exists(os.path.join(ROOT, "backend", "static", "vendor", "ort", "ort.wasm.min.mjs"))
GLASSES = {"width": 640, "height": 360}

needs_text = pytest.mark.skipif(not (HAS_TEXT and HAS_RUNTIME and sc.fonts_available()),
                                reason="text reader models missing: python tools/fetch_assist_models.py")
needs_speech = pytest.mark.skipif(not (HAS_SPEECH and HAS_RUNTIME), reason="speech model missing: python tools/fetch_assist_models.py")
needs_resistor = pytest.mark.skipif(not (HAS_RESISTOR and HAS_RUNTIME), reason="resistor model missing: python -m backend.resistor.export")

STATE = "() => window.assist.state()"

# what a line from the text reader looks like, for tests that skip the camera
LINES_JS = """
const line = (text, size = 60, x = 100, y = 100, conf = 0.98) =>
  ({ text, conf, lineHeight: size, x, y, w: text.length * size * 0.55, h: size, cx: x + text.length * size * 0.275, cy: y + size / 2 });
"""


def detail(page):
    return page.evaluate(STATE)["detail"]


# ------------------------------------------------------------ rule engine ---
def test_label_matching_is_conservative():
    with browser_page() as (page, _):
        res = page.evaluate("""async () => {
            %s
            const { SafetyKB } = await import('/static/js/assist/kb.js');
            const kb = await SafetyKB.load();
            const ids = (...texts) => kb.findInLines(texts.map((t, i) => line(t, 60, 100, 100 + i * 80))).map((h) => h.sub.id).sort();
            const tags = (...texts) => [...kb.profile(ids(...texts)).tags].sort();
            return {
              plain: ids('ULTRA BLEACH'),
              warning: ids('Do not mix with bleach'),
              never: ids('Never use with ammonia'),
              almond: ids('ALMOND MILK'),
              peanut: ids('Crunchy Peanut Butter'),
              beach: ids('BEACH PARTY'),
              digit: ids('Hydr0gen Peroxide 3%%'),
              fuzzy: kb.findInLines([line('AMMONlA')]).map((h) => [h.sub.id, h.exact]),
              wrapped: ids('ENERGY', 'DRINK'),
              formulas: ids('NaOH 2M', 'H202', 'HCI'),
              shortFormula: ids('Na', 'K'),
              alcoholFree: tags('ALCOHOL FREE', 'BEER'),
              sanitiser: tags('Isopropyl Alcohol 70%%'),
              splitLabel: kb.profile(ids('ISOPROPYL', '91%%', 'ALCOHOL')).subs.map((s) => s.id),
              splitKinds: kb.profile(ids('ISOPROPYL', '91%%', 'ALCOHOL')).kinds,
              decafCola: tags('Caffeine Free Cola'),
              glutenFree: kb.profile(ids('Gluten Free Bread')).allergens,
              noodles: tags('Chicken Noodles'),
              rawChicken: tags('Chicken Breast'),
              peanutFree: ids('PEANUT FREE'),
            };
        }""" % LINES_JS)
        assert res["plain"] == ["bleach"]
        assert res["warning"] == [] and res["never"] == [], "a warning about X is not X"
        assert res["almond"] == ["almond-milk"], "the longest phrase wins: almond milk is not dairy milk"
        assert res["peanut"] == ["peanuts"]
        assert res["beach"] == [], "a short word must match exactly"
        assert res["digit"] == ["hydrogen-peroxide"]
        assert res["fuzzy"] == [["ammonia", False]], "one misread letter in a long word is tolerated, and flagged"
        assert res["wrapped"] == ["energy-drink"]
        assert res["formulas"] == ["drain-cleaner", "hydrochloric-acid", "hydrogen-peroxide"]
        assert res["shortFormula"] == [], "two-letter printed fragments are not accepted as formulas"
        assert "alcohol" not in res["alcoholFree"]
        assert "alcohol" not in res["sanitiser"] and "solvent-alcohol" in res["sanitiser"]
        # the two words read as separate lines: still one chemical, not also a drink
        assert res["splitLabel"] == ["rubbing-alcohol"] and res["splitKinds"] == ["chemical"]
        assert "caffeine" not in res["decafCola"] and "carbonated" in res["decafCola"]
        assert res["glutenFree"] == []
        assert "maybe-raw" not in res["noodles"] and "maybe-raw" in res["rawChicken"]
        assert res["peanutFree"] == []
        assert not page.problems, page.problems


def test_verdicts_and_reasons():
    with browser_page() as (page, _):
        res = page.evaluate("""async () => {
            const { SafetyKB } = await import('/static/js/assist/kb.js');
            const kb = await SafetyKB.load();
            const judge = (a, b) => { const v = kb.judge(kb.profile(a), kb.profile(b));
              return { level: v.level, rules: v.findings.map((f) => f.id), why: v.findings[0]?.why || '' }; };
            return {
              bleachAmmonia: judge(['bleach'], ['ammonia']),
              reversed: judge(['ammonia'], ['bleach']),
              drains: judge(['drain-cleaner'], ['toilet-cleaner']),
              milkIron: judge(['milk'], ['iron-supplement']),
              vodkaRedBull: judge(['alcohol-drink'], ['energy-drink']),
              bleachMilk: judge(['bleach'], ['milk']),
              bleachVinegar: judge(['bleach'], ['vinegar']),
              milkCoffee: judge(['milk'], ['coffee']),
              nonAlcoholic: judge(['alcohol-drink', 'alcohol-free'], ['energy-drink']),
              headline: kb.judge(kb.profile(['milk']), kb.profile(['coffee'])).headline,
            };
        }""")
        assert res["bleachAmmonia"]["level"] == "danger" and "chloramine" in res["bleachAmmonia"]["why"]
        assert res["reversed"] == res["bleachAmmonia"], "order must not matter"
        # one rule per group: the violent strong-acid/strong-base rule, not also the mild one
        assert res["drains"]["rules"] == ["strong-acid-base"]
        assert res["milkIron"]["level"] == "info"
        assert res["vodkaRedBull"]["rules"][0] == "alcohol-caffeine"
        assert res["bleachMilk"]["rules"] == ["chemical-drink"]
        assert res["bleachVinegar"]["rules"][0] == "bleach-acid" and res["bleachVinegar"]["level"] == "danger"
        assert res["milkCoffee"] == {"level": "none", "rules": [], "why": ""}
        assert res["nonAlcoholic"]["level"] == "none"
        assert res["headline"] == "No known problem"


def test_tray_remembers_two_items_shown_one_after_the_other():
    with browser_page() as (page, _):
        res = page.evaluate("""async () => {
            const { SafetyKB } = await import('/static/js/assist/kb.js');
            const { ItemTray, splitClusters } = await import('/static/js/assist/safety.js');
            %s
            const kb = await SafetyKB.load();
            const sub = (id) => kb.subs.get(id);
            const box = { x: 400, y: 200, w: 300, h: 200 };
            const seen = (id, exact = true, conf = 0.97) => [{ box, hits: [{ sub: sub(id), exact, conf, size: 60 }] }];
            const names = (tray) => tray.items.map((it) => it.slot + ':' + it.profile.subs.map((s) => s.id).join('+'));
            const out = {};

            let tray = new ItemTray(kb);
            tray.update(seen('ammonia', false, 0.7), 0);
            out.oneShakyRead = names(tray);                    // a single uncertain read is not believed
            tray.update(seen('ammonia', false, 0.7), 800);
            out.twoShakyReads = names(tray);

            tray = new ItemTray(kb);
            tray.update(seen('bleach'), 0);
            tray.update(seen('bleach'), 900);                  // same package again: still one item
            out.same = names(tray);
            tray.update([], 2000);                             // package lowered
            tray.update(seen('ammonia'), 5000);                // a different package raised in the same place
            out.sequence = names(tray);
            tray.update(seen('vinegar'), 9000);                // a third replaces the one seen longest ago
            out.third = names(tray);
            tray.update([], 9000 + 31000);
            out.expired = names(tray);

            tray = new ItemTray(kb);                           // two identical packs side by side are two items
            tray.update([{ box: { x: 100, y: 200, w: 300, h: 200 }, hits: seen('paracetamol')[0].hits },
                         { box: { x: 800, y: 200, w: 300, h: 200 }, hits: seen('paracetamol')[0].hits }], 0);
            out.twins = names(tray);

            const groups = splitClusters([line('BLEACH', 100, 80, 300), line('ULTRA', 60, 140, 200),
                                          line('AMMONIA', 80, 760, 300), line('Glass Cleaner', 70, 720, 200)], 1280);
            out.split = groups.map((g) => g.map((l) => l.text));
            out.single = splitClusters([line('WHOLE', 80, 500, 200), line('MILK', 120, 480, 300)], 1280).length;
            return out;
        }""" % LINES_JS)
        assert res["oneShakyRead"] == [] and res["twoShakyReads"] == ["A:ammonia"]
        assert res["same"] == ["A:bleach"]
        assert res["sequence"] == ["A:bleach", "B:ammonia"], "a different label in the same place is a second item"
        assert res["third"] == ["A:vinegar", "B:ammonia"]
        assert res["expired"] == []
        assert res["twins"] == ["A:paracetamol", "B:paracetamol"]
        assert res["split"] == [["ULTRA", "BLEACH"], ["Glass Cleaner", "AMMONIA"]]
        assert res["single"] == 1


# ------------------------------------------------------------ safety mode ---
def _wait_verdict(page):
    page.wait_for_function("() => window.assist?.state().detail?.verdict", timeout=120000)
    return detail(page)


@needs_text
def test_safety_two_cleaners_on_the_glasses_screen():
    scene = sc.packages(((30, 90, 200), [("ULTRA", 60, True), ("BLEACH", 120, True), ("Kills 99.9% of germs", 34, False)]),
                        ((250, 250, 250), [("Glass Cleaner", 76, True), ("with AMMONIA", 64, True),
                                           ("Streak free shine", 34, False)], (20, 60, 160)),
                        name="assist_cleaners.png")
    with browser_page(f"/assist?mode=safety&hand=0&src={scene}", viewport=GLASSES) as (page, _):
        d = _wait_verdict(page)
        assert [i["subs"] for i in d["items"]] == [["bleach"], ["glass-cleaner", "ammonia"]], d["items"]
        assert d["verdict"] == {"level": "danger", "rules": ["bleach-ammonia"]}
        assert page.locator("#verdict").get_attribute("data-level") == "danger"
        text = page.locator("#verdict").inner_text()
        assert "DO NOT COMBINE" in text.upper() and "toxic gas" in text and "chloramine" in text
        # everything on screen stays inside the 640x360 display
        overflow = page.evaluate("""() => [...document.querySelectorAll('#top, #actions, .slots, #verdict')]
            .filter((el) => { const r = el.getBoundingClientRect(); return r.left < 0 || r.top < 0 || r.right > innerWidth || r.bottom > innerHeight; })
            .map((el) => el.id || el.className)""")
        assert overflow == [], overflow
        # the reason is one press away, with its source
        page.click("[data-act=details]")
        sheet = page.locator("#sheetBody").inner_text()
        assert "Never mix" in sheet and "Source: CDC" in sheet and "not medical or professional advice" in sheet
        assert not page.problems, page.problems


@needs_text
def test_safety_food_and_medicine_pairs():
    cases = [
        (((240, 240, 235), [("Fresh", 60, False), ("MILK", 130, True), ("Full cream", 40, False)], (20, 70, 150)),
         ((170, 30, 40), [("IRON", 90, True), ("TABLETS", 80, True), ("Ferrous sulfate 200 mg", 34, False)]),
         "info", "iron-calcium"),
        (((20, 20, 24), [("VODKA", 120, True), ("40% ABV", 60, False)]),
         ((20, 60, 180), [("ENERGY", 96, True), ("DRINK", 96, True), ("250 ml", 40, False)]),
         "caution", "alcohol-caffeine"),
        (((230, 150, 20), [("ORANGE", 90, True), ("JUICE", 90, True)], (60, 30, 0)),
         ((120, 70, 30), [("COFFEE", 110, True), ("Rich roast", 44, False)]),
         "none", None),
    ]
    for i, (left, right, level, rule) in enumerate(cases):
        scene = sc.packages(left, right, name=f"assist_food_{i}.png")
        with browser_page(f"/assist?mode=safety&hand=0&src={scene}", viewport=GLASSES) as (page, _):
            d = _wait_verdict(page)
            assert d["verdict"]["level"] == level, (i, d)
            assert (d["verdict"]["rules"] or [None])[0] == rule, (i, d)
            if level == "none":
                assert "not a guarantee" in page.locator("#verdict").inner_text()
            assert not page.problems, page.problems


@needs_text
def test_safety_allergy_flag_and_single_item():
    scene = sc.packages(((150, 90, 40), [("Crunchy", 60, False), ("PEANUT", 110, True), ("BUTTER", 110, True)]),
                        name="assist_peanut.png")
    with browser_page("/static/blank.html", viewport=GLASSES) as (page, base):
        page.evaluate("() => localStorage.setItem('cvision-assist', JSON.stringify({ allergens: ['peanuts'] }))")
        page.goto(base + f"/assist?mode=safety&hand=0&src={scene}")
        page.wait_for_function("() => window.assist?.state().detail?.items?.length === 1", timeout=120000)
        d = detail(page)
        assert d["items"][0]["subs"] == ["peanuts"] and d["items"][0]["allergens"] == ["peanuts"]
        assert d["verdict"] is None
        assert "Your allergy: peanuts" in page.locator("#slotA").inner_text()
        assert "ONE ITEM SO FAR" in page.locator("#verdict").inner_text().upper()
        # the allergy list is editable with the keyboard alone
        page.click("[data-act=allergy]")
        assert page.evaluate("() => document.activeElement.dataset.allergen") == "milk"
        page.keyboard.press("Enter")
        assert page.evaluate("() => JSON.parse(localStorage.getItem('cvision-assist')).allergens.sort()") == ["milk", "peanuts"]
        # Clear empties the tray
        page.keyboard.press("Escape")
        page.click("[data-act=clear]")
        assert detail(page)["items"] == []
        assert not page.problems, page.problems


@needs_text
def test_safety_reads_handwritten_formulas():
    fixture = os.path.join(ROOT, "tests", "fixtures", "handwriting_cards.png")
    os.makedirs(sc.OUT_DIR, exist_ok=True)
    shutil.copy(fixture, os.path.join(sc.OUT_DIR, "handwriting_cards.png"))
    with browser_page("/assist?mode=safety&src=/static/test/handwriting_cards.png",
                      viewport={"width": 412, "height": 915}) as (page, _):
        page.wait_for_function("() => window.assist?.state().detail?.items?.length === 2", timeout=120000)
        d = detail(page)
        assert sorted(i["subs"][0] for i in d["items"]) == ["salt", "water"], d["items"]
        assert sorted(i["formula"] for i in d["items"] if i["formula"]) in (["H2O", "NaCl"], ["NaCl"], ["H2O"]), d["items"]
        assert d["verdict"]["level"] == "none"
        assert not page.problems, page.problems


# ----------------------------------------------------- navigation / shell ---
def test_touchpad_navigation_and_lifecycle():
    """Everything is reachable with arrow keys, Enter and Back; the camera is released on pause."""
    with browser_page("/assist", viewport=GLASSES) as (page, _):
        page.wait_for_function("() => window.assist && document.activeElement?.dataset?.mode")
        assert page.evaluate("() => document.activeElement.dataset.mode") == "safety"
        page.keyboard.press("ArrowRight")
        assert page.evaluate("() => document.activeElement.dataset.mode") == "electronics"
        page.keyboard.press("ArrowLeft")
        page.keyboard.press("Enter")                              # start Safety with the fake camera
        page.wait_for_function("() => window.assist.running && window.assist.geo")
        page.wait_for_function("() => window.assist.state().mode === 'safety'", timeout=120000)
        s = page.evaluate(STATE)
        assert s["actions"] == ["details", "clear", "allergy", "online"]
        assert s["focus"] in ("details", "pause"), s["focus"]     # the cursor landed on a control
        track = "() => document.getElementById('cam').srcObject.getVideoTracks()[0]"
        page.evaluate(f"() => {{ window.oldTrack = ({track})(); }}")

        # walk the controls: every one is reachable, and the walk wraps round
        seen = []
        for _ in range(9):
            page.keyboard.press("ArrowRight")
            seen.append(page.evaluate(STATE)["focus"])
        assert {"safety", "electronics", "access", "pause", "details", "clear", "allergy", "online"} <= set(seen), seen

        # open a sheet with Enter, close it with Back; Back again pauses
        page.focus("[data-act=details]")
        page.keyboard.press("Enter")
        assert page.evaluate(STATE)["sheet"] is True
        page.keyboard.press("Escape")
        assert page.evaluate(STATE)["sheet"] is False and page.evaluate(STATE)["running"] is True
        page.keyboard.press("Escape")
        assert page.evaluate("() => !window.assist.running && oldTrack.readyState === 'ended'")
        assert page.locator("#start").is_visible()
        assert page.evaluate("() => document.activeElement.dataset.mode") == "safety"
        errors = [p for p in page.problems if "favicon" not in p]
        assert not errors, errors


def test_browser_back_button_closes_instead_of_leaving():
    with browser_page("/assist", viewport=GLASSES) as (page, _):
        page.wait_for_function("() => window.assist && document.activeElement?.dataset?.mode")
        page.click("#modes [data-mode=safety]")
        page.wait_for_function("() => window.assist.state().mode === 'safety'", timeout=120000)
        page.click("[data-act=details]")
        page.go_back()
        page.wait_for_function("() => !window.assist.state().sheet")
        assert page.evaluate(STATE)["running"] is True and "/assist" in page.url
        page.go_back()
        page.wait_for_function("() => !window.assist.running")
        assert "/assist" in page.url and page.locator("#start").is_visible()


def test_pausing_pending_camera_permission_allows_a_new_start():
    with browser_page("/assist", viewport=GLASSES) as (page, _):
        page.wait_for_function("() => window.assist")
        result = page.evaluate("""async () => {
            let grantFirst, rejectSecond, calls = 0, stopped = 0;
            navigator.mediaDevices.getUserMedia = () => ++calls === 1
              ? new Promise(resolve => grantFirst = resolve)
              : new Promise((resolve, reject) => rejectSecond = reject);
            const first = assist.start('safety');
            assist.stop();
            const second = assist.start('electronics');
            grantFirst({getTracks: () => [{stop: () => stopped++}]});
            await first;
            const stillStarting = assist.starting;
            rejectSecond(new Error('camera unavailable'));
            await second;
            return {calls, stopped, stillStarting, running: assist.running, starting: assist.starting};
        }""")
        assert result == {"calls": 2, "stopped": 1, "stillStarting": True, "running": False, "starting": False}


def test_pausing_during_wake_lock_releases_the_late_lock():
    with browser_page("/assist", viewport=GLASSES) as (page, _):
        page.wait_for_function("() => window.assist")
        result = page.evaluate("""async () => {
            let grant, requested, released = 0, modeStarts = 0;
            const requestStarted = new Promise(resolve => requested = resolve);
            Object.defineProperty(navigator, 'wakeLock', {configurable: true, value: {
              request: () => { requested(); return new Promise(resolve => grant = resolve); }
            }});
            assist.setMode = async () => { modeStarts++; };
            const start = assist.start('safety');
            await requestStarted;
            const track = document.getElementById('cam').srcObject.getVideoTracks()[0];
            assist.stop();
            grant({release: async () => { released++; }});
            await start;
            return {released, modeStarts, running: assist.running, track: track.readyState, lock: !!assist.wakeLock};
        }""")
        assert result == {"released": 1, "modeStarts": 0, "running": False, "track": "ended", "lock": False}


def test_missing_models_show_a_message_not_a_blank_screen():
    with browser_page("/static/blank.html", viewport=GLASSES) as (page, base):
        page.route("**/static/models/assist/**", lambda route: route.fulfill(status=404, body="missing"))
        page.goto(base + "/assist?mode=safety&hand=0&autostart=1")
        page.wait_for_function("() => document.querySelector('#mode')?.innerText.includes('could not start')", timeout=120000)
        text = page.locator("#mode").inner_text()
        assert "fetch_assist_models" in text
        # the app is still usable: another tab can be opened
        assert page.evaluate(STATE)["running"] is True


def test_phone_portrait_layout_fits():
    with browser_page("/assist?mode=safety&hand=0&autostart=1", viewport={"width": 390, "height": 844}) as (page, _):
        page.wait_for_function("() => window.assist?.state().mode === 'safety'", timeout=120000)
        overflow = page.evaluate("""() => [...document.querySelectorAll('#top, #tabs, #actions, .slots, #verdict, .act, .tab')]
            .filter((el) => { const r = el.getBoundingClientRect(); return r.left < -1 || r.right > innerWidth + 1 || r.bottom > innerHeight + 1; })
            .map((el) => el.id || el.className)""")
        assert overflow == [], overflow


# ------------------------------------------------------- electronics mode ---
def test_electronics_refreshes_shared_reader_and_discards_roundtrip_results():
    with browser_page() as (page, _):
        result = page.evaluate("""async () => {
            const { ElectronicsMode } = await import('/static/js/assist/electronics.js');
            let finishChip, finishResistor, readerLoads = 0;
            const reader = {due: () => true, read: () => new Promise(resolve => finishChip = resolve)};
            const app = {opts: {}, generation: 1, busy() {}, updateAction() {}, stats: {}, video: {srcObject: null}, motion: {mark() {}},
              textReader: async () => { readerLoads++; return reader; }};
            const mode = new ElectronicsMode(app); app.mode = mode;
            mode.parts = [{id: '555', regex: /^NE555$/, name: 'Timer'}];
            mode.resistor = {read: () => new Promise(resolve => finishResistor = resolve)};
            mode.reader = {disposed: true, due: () => false};
            mode.box = () => ({sx: 0, sy: 0, sw: 100, sh: 100});
            await mode.setSub('chip');
            const refreshed = mode.reader === reader;
            mode.tickChip(1000);
            await mode.setSub('resistor');
            await mode.setSub('chip');
            finishChip({totalMs: 1, lines: [{text: 'NE555', conf: 1, lineHeight: 30}]});
            await new Promise(resolve => requestAnimationFrame(resolve));
            await mode.setSub('resistor');
            mode.tickResistor(2000);
            mode.leave();
            await mode.setSub('resistor', true);
            finishResistor({bands: ['brown', 'black', 'red', 'gold'], conf: 1});
            await new Promise(resolve => requestAnimationFrame(resolve));
            return {refreshed, readerLoads, reads: mode.reads, chip: mode.chip, resistor: mode.value};
        }""")
        assert result == {"refreshed": True, "readerLoads": 2, "reads": 0, "chip": None, "resistor": None}


@needs_text
def test_chip_lookup_and_pinout():
    scene = sc.chip("SN74HC595N", "2148 G4", name="assist_chip.png")
    with browser_page(f"/assist?mode=electronics&sub=chip&src={scene}", viewport=GLASSES) as (page, _):
        page.wait_for_function("() => window.assist?.state().detail?.chip", timeout=120000)
        d = detail(page)
        assert d["chip"]["id"] == "74595" and d["chip"]["marking"] == "SN74HC595N"
        assert page.evaluate(STATE)["zoom"] > 1.5, "small parts are shown magnified"
        page.click("[data-act=pinout]")
        sheet = page.locator("#sheetBody")
        assert sheet.locator("svg text").count() >= 32          # 16 numbers + 16 names
        names = page.evaluate("() => [...document.querySelectorAll('#sheetBody svg text')].map((t) => t.textContent)")
        assert "SER (data)" in names and "VCC" in names and "GND" in names and "OE" in names
        assert "Tie" in sheet.inner_text()
        assert not page.problems, page.problems


def test_chip_markings_with_misreads_and_the_parts_browser():
    with browser_page("/assist?mode=electronics&sub=chip&autostart=1", viewport=GLASSES) as (page, _):
        page.wait_for_function("() => window.assist?.state().mode === 'electronics' && window.assist.mode.parts", timeout=120000)
        res = page.evaluate("""async () => {
            const { findPart, pinoutSvg } = await import('/static/js/assist/electronics.js');
            const parts = window.assist.mode.parts;
            const id = (...texts) => { const r = findPart(parts, texts.map((t, i) => ({ text: t, conf: 0.95, lineHeight: 50 - i })));
              return r ? r.part.id + '/' + r.groups.join(',') : null; };
            return { spaced: id('LM 358 N'), letterO: id('NE555P', 'O2A7'), o_for_zero: id('CD4O17BE'), regulator: id('L7805CV'),
              small: id('78L05'), date: id('2148', 'MALAYSIA'), none: id('HELLO WORLD'), transistor: id('BC547B'),
              svgs: parts.map((p) => pinoutSvg(p).length > 400).every(Boolean), count: parts.length };
        }""")
        assert res["spaced"] == "358/" and res["letterO"] == "555/" and res["o_for_zero"] == "4017/"
        assert res["regulator"] == "78xx/05" and res["small"] == "78Lxx/05" and res["transistor"] == "BC547/"
        assert res["date"] is None and res["none"] is None
        assert res["svgs"] and res["count"] >= 55
        # browse the list and open a three-legged part with the keyboard only
        page.click("[data-act=browse]")
        assert page.locator("#sheetBody .row").count() == res["count"]
        page.locator("#sheetBody [data-part='78xx']").focus()
        page.keyboard.press("Enter")
        assert "78xx positive voltage regulator" in page.locator("#sheetTitle").inner_text()
        labels = page.evaluate("() => [...document.querySelectorAll('#sheetBody svg text')].map((t) => t.textContent)")
        assert [x for x in labels if x in ("IN", "GND", "OUT")] == ["IN", "GND", "OUT"]
        assert not page.problems, page.problems


@needs_resistor
def test_resistor_reader_on_drawn_resistors():
    cases = [(["yellow", "violet", "red", "gold"], 4700, 5, 3), (["brown", "black", "orange", "gold"], 10000, 5, 5),
             (["red", "red", "brown", "gold"], 220, 5, 8)]
    for bands, ohms, tol, seed in cases:
        scene = sc.resistor_frame(bands, seed=seed, name=f"assist_res_{ohms}.png")
        with browser_page(f"/assist?mode=electronics&sub=resistor&src={scene}", viewport=GLASSES) as (page, _):
            page.wait_for_function("() => window.assist?.state().detail?.resistor", timeout=120000)
            d = detail(page)
            assert d["resistor"]["ohms"] == ohms and d["resistor"]["tolerance"] == tol, (bands, d)
            assert d["resistor"]["bands"] == bands
            assert page.locator("#eResult .bands i:not(.gap)").count() == len(bands)
            assert not page.problems, page.problems


@needs_resistor
def test_resistor_reader_stays_quiet_with_no_resistor():
    scene = sc.board(["Nothing to see", "just a page of text"], name="assist_no_resistor.png")
    with browser_page(f"/assist?mode=electronics&sub=resistor&src={scene}", viewport=GLASSES) as (page, _):
        page.wait_for_function("() => window.assist?.state().detail?.reads >= 8", timeout=120000)
        assert detail(page)["resistor"] is None
        assert "Fill the box" in page.locator("#eHint").inner_text()


# ------------------------------------------------------------ access mode ---
def test_native_speech_availability_and_lifecycle_bridge():
    with browser_page() as (page, _):
        result = page.evaluate("""async () => {
            const { AccessMode } = await import('/static/js/assist/access.js');
            let available = false, stopped = 0, actions = [], updates = 0;
            const spoken = [];
            window.AssistNative = {isSpeechAvailable: () => available,
              speak: text => spoken.push(text), stopSpeaking: () => stopped++};
            const app = {opts: {sub: 'read'}, generation: 1, busy() {}, updateAction() {},
              setActions: next => { actions = next.map(a => a.id); updates++; }};
            const mode = new AccessMode(app); app.mode = mode;
            const initiallyAvailable = mode.voice;
            mode.speak(['Unavailable']);
            available = true;
            window.dispatchEvent(new Event('assist-native-speech'));
            const nowAvailable = mode.voice && actions.includes('speak');
            mode.speak(['Title', 'First sentence.']);
            app.mode = null;
            available = false;
            window.dispatchEvent(new Event('assist-native-speech'));
            mode.speak(['Unavailable again']);
            mode.leave();
            return {initiallyAvailable, nowAvailable, finalAvailable: mode.voice, spoken, updates, stopped};
        }""")
        assert result == {"initiallyAvailable": False, "nowAvailable": True, "finalAvailable": False,
                          "spoken": ["Title\n\nFirst sentence."], "updates": 1, "stopped": 2}


@needs_speech
def test_captions_from_speech():
    shutil.copy(os.path.join(ROOT, "tests", "fixtures", "speech_sample.wav"), os.path.join(sc.OUT_DIR, "speech_sample.wav"))
    with browser_page("/assist?mode=access&sub=captions&audio=/static/test/speech_sample.wav&autostart=1",
                      viewport=GLASSES) as (page, _):
        page.wait_for_function("() => window.assist?.state().detail?.captions?.finals?.length >= 2", timeout=180000)
        d = detail(page)
        heard = " ".join(d["captions"]["finals"]).lower()
        for word in ("plants", "glucose", "carbon dioxide", "water", "sunlight"):
            assert word in heard, heard
        assert "glucose" in page.locator(".captions .text").inner_text()
        # leaving Captions lets go of the speech model and the microphone
        page.click("#tabs [data-mode=safety]")
        page.wait_for_function("() => window.assist.state().mode === 'safety'", timeout=120000)
        assert page.evaluate("() => { const a = window.assist.modes.get('access'); return !a.worker && !a.stream && !a.listening; }")
        assert not page.problems, page.problems


def test_captions_microphone_is_opened_and_released():
    """With the browser's fake microphone: the mic opens only in Captions and closes on leaving it."""
    if not (HAS_SPEECH and HAS_RUNTIME):
        pytest.skip("speech model missing")
    with browser_page("/assist?mode=access&sub=captions&autostart=1", viewport=GLASSES) as (page, _):
        page.wait_for_function("() => window.assist?.state().detail?.captions?.listening", timeout=180000)
        assert page.evaluate("() => window.assist.mode.stream.getAudioTracks()[0].readyState") == "live"
        page.evaluate("() => { window.micTrack = window.assist.mode.stream.getAudioTracks()[0]; }")
        page.click("[data-act=read]")
        page.wait_for_function("() => window.assist.state().detail.sub === 'read'")
        assert page.evaluate("() => micTrack.readyState") == "ended"
        assert page.evaluate("() => !window.assist.mode.worker"), "Read releases the speech model too"
        errors = [p for p in page.problems if "favicon" not in p]
        assert not errors, errors


def test_captions_release_a_late_microphone_permission():
    """Leaving while permission is pending must stop a subsequently granted stream."""
    with browser_page() as (page, _):
        result = page.evaluate("""async () => {
            const { AccessMode } = await import('/static/js/assist/access.js');
            const workers = [];
            window.Worker = class {
              constructor() { workers.push(this); }
              postMessage(m) { if (m.type === 'init') queueMicrotask(() => this.onmessage({data: {type: 'ready'}})); }
              terminate() { this.terminated = true; }
            };
            let grant, requested;
            const permissionRequested = new Promise(resolve => requested = resolve);
            navigator.mediaDevices.getUserMedia = () => {
              requested(); return new Promise(resolve => grant = resolve);
            };
            const app = {opts: {}, generation: 1, busy() {}, updateAction() {}, textReader: async () => ({})};
            const mode = new AccessMode(app); app.mode = mode;
            const start = mode.startCaptions();
            await permissionRequested;
            await mode.setSub('read');
            let stopped = 0;
            grant({getTracks: () => [{stop: () => stopped++}]});
            await start;
            return {stopped, listening: mode.listening, stream: !!mode.stream,
              worker: !!mode.worker, terminated: workers.every(w => w.terminated), error: mode.error};
        }""")
        assert result == {"stopped": 1, "listening": False, "stream": False,
                          "worker": False, "terminated": True, "error": None}


def test_captions_cancel_pending_model_and_allow_restart():
    """Back cancels model loading immediately; returning can start a fresh worker."""
    with browser_page() as (page, _):
        result = page.evaluate("""async () => {
            const { AccessMode } = await import('/static/js/assist/access.js');
            const workers = [];
            window.Worker = class {
              constructor() { workers.push(this); }
              postMessage() {}
              terminate() { this.terminated = true; }
            };
            let micRequests = 0;
            navigator.mediaDevices.getUserMedia = async () => { micRequests++; throw new Error('unexpected microphone'); };
            const app = {opts: {}, generation: 1, busy() {}, updateAction() {}};
            const mode = new AccessMode(app); app.mode = mode;
            const first = mode.startCaptions();
            mode.leave();
            await first;
            const second = mode.startCaptions();
            mode.leave();
            await second;
            return {workers: workers.length, terminated: workers.every(w => w.terminated), micRequests,
              starting: mode.startingCaptions, loading: !!mode.captionLoader, error: mode.error};
        }""")
        assert result == {"workers": 2, "terminated": True, "micRequests": 0,
                          "starting": False, "loading": False, "error": None}


def test_crashed_caption_worker_releases_microphone():
    with browser_page() as (page, _):
        result = page.evaluate("""async () => {
            const { AccessMode } = await import('/static/js/assist/access.js');
            let worker, stopped = 0;
            window.Worker = class {
              constructor() { worker = this; }
              postMessage(m) { if (m.type === 'init') queueMicrotask(() => this.onmessage({data: {type: 'ready'}})); }
              terminate() { this.terminated = true; }
            };
            const app = {opts: {}, generation: 1, busy() {}, updateAction() {}};
            const mode = new AccessMode(app); app.mode = mode;
            mode.openMic = async () => { mode.stream = {getTracks: () => [{stop: () => stopped++}]}; };
            await mode.startCaptions();
            worker.onerror({message: 'Worker crashed'});
            return {stopped, terminated: worker.terminated, error: mode.error, listening: mode.listening,
              stream: !!mode.stream, worker: !!mode.worker};
        }""")
        assert result == {"stopped": 1, "terminated": True, "error": "Worker crashed",
                          "listening": False, "stream": False, "worker": False}


def test_read_result_is_discarded_when_switching_to_captions():
    with browser_page() as (page, _):
        result = page.evaluate("""async () => {
            const { AccessMode } = await import('/static/js/assist/access.js');
            let finish, spoken = 0;
            const app = {opts: {}, generation: 1, busy() {}, updateAction() {}, stats: {}, geo: {sx: 0, sy: 0, sw: 100, sh: 100}};
            const mode = new AccessMode(app); app.mode = mode;
            mode.sub = 'read';
            mode.reader = {read: () => new Promise(resolve => finish = resolve)};
            mode.startCaptions = async () => {};
            mode.speak = () => { spoken++; };
            const read = mode.readNow();
            await mode.setSub('captions');
            finish({totalMs: 1, lines: [{text: 'Old text', conf: 1, lineHeight: 20, length: 80, cx: 40, cy: 10}]});
            await read;
            return {spoken, open: mode.readerOpen, text: mode.text, sub: mode.sub};
        }""")
        assert result == {"spoken": 0, "open": False, "text": [], "sub": "captions"}


@needs_text
def test_read_aloud_shows_the_text_in_order():
    scene = sc.board(["Photosynthesis", "Plants make glucose from", "carbon dioxide and water", "using energy from sunlight."],
                     name="assist_board.png")
    with browser_page(f"/assist?mode=access&sub=read&src={scene}", viewport=GLASSES) as (page, _):
        page.wait_for_function("() => window.assist?.state().detail?.sub === 'read' && window.assist.mode.reader", timeout=120000)
        page.evaluate("() => { window.spoken = []; if (window.speechSynthesis) speechSynthesis.speak = (u) => spoken.push(u.text); }")
        page.focus("[data-act=read]")
        page.keyboard.press("Enter")
        page.wait_for_function("() => window.assist.state().detail.read.open", timeout=120000)
        d = detail(page)
        assert d["read"]["text"] == ["Photosynthesis", "Plants make glucose from carbon dioxide and water using energy from sunlight."]
        assert "Photosynthesis" in page.locator("#readerBox").inner_text()
        if d["voice"]:
            assert page.evaluate("() => spoken") == d["read"]["text"], "the text is handed to the device's voice"
        page.keyboard.press("Escape")                              # Back closes the text, not the app
        assert detail(page)["read"]["open"] is False and page.evaluate(STATE)["running"] is True
        assert not page.problems, page.problems


# ---------------------------------------------------------------- barcodes ---
def test_safety_clear_discards_pending_label_and_barcode_results():
    with browser_page() as (page, _):
        result = page.evaluate("""async () => {
            const { SafetyMode } = await import('/static/js/assist/safety.js');
            let labelReady, barcodeReady, judged = 0;
            const app = {opts: {}, settings: {}, generation: 1, geo: {sx: 0, sy: 0, sw: 100, sh: 100},
              motion: {mark() {}}, toast() {}};
            const mode = new SafetyMode(app); app.mode = mode;
            mode.reader = {read: () => new Promise(resolve => labelReady = resolve)};
            mode.barcodes.scan = () => '2000000000015';
            mode.book = {find: () => new Promise(resolve => barcodeReady = resolve)};
            mode.judge = () => { judged++; };
            const pass = mode.pass(10000);
            mode.lookForBarcode(10000);
            mode.clear();
            labelReady({totalMs: 1, lines: []});
            barcodeReady({name: 'Old product'});
            await pass;
            return {judged, passes: mode.passes, products: mode.products, codes: mode.codes.size};
        }""")
        assert result == {"judged": 0, "passes": 0, "products": [], "codes": 0}


def _scanline(code, module=3.0, margin=40, blur=1.0, noise=6, seed=0, flip=False):
    """Grey levels along a line through a drawn barcode, as a camera would see it."""
    import numpy as np
    bits = sc.ean13_bits(code)
    n = int(len(bits) * module) + 2 * margin
    x = (np.arange(n) - margin) / module
    line = np.full(n, 225.0)
    inside = (x >= 0) & (x < len(bits))
    line[inside] = np.where(np.array([bits[int(v)] for v in x[inside]]) == "1", 28.0, 225.0)
    if blur:
        k = np.exp(-0.5 * (np.arange(-4, 5) / blur) ** 2)
        line = np.convolve(np.pad(line, 4, mode="edge"), k / k.sum(), mode="valid")
    line = line + np.random.default_rng(seed).normal(0, noise, n)
    return [int(v) for v in np.clip(line[::-1] if flip else line, 0, 255)]


def test_barcode_lines_decode_and_bad_ones_do_not():
    pytest.importorskip("numpy")
    codes = [sc.ean13_check(c) for c in ("200000000001", "890103086527", "501234567890", "012345678905", "400638133393",
                                         "762221044961", "978030640615", "300000000000", "111111111111", "690123456789")]
    lines = []
    for i, code in enumerate(codes):
        lines.append({"want": code, "line": _scanline(code, module=[2.0, 3.0, 4.6][i % 3], seed=i, flip=i % 2 == 1)})
    wrong = codes[0][:12] + str((int(codes[0][12]) + 3) % 10)             # same bars as a real code, wrong check digit
    lines.append({"want": None, "line": _scanline(wrong, seed=50)})
    import numpy as np
    rng = np.random.default_rng(9)
    for _ in range(40):                                                   # stripes that are not a barcode
        runs = rng.integers(2, 14, 90)
        lines.append({"want": None, "line": [int(v) for v in np.repeat(np.tile([225, 30], 45), runs)]})
    lines.append({"want": None, "line": [int(v) for v in rng.integers(0, 255, 600)]})
    with browser_page() as (page, _):
        got = page.evaluate("""async (lines) => {
            const { scanLine, validEan13 } = await import('/static/js/assist/barcode.js');
            return { read: lines.map((l) => scanLine(Uint8Array.from(l.line))), valid: [validEan13('2000000000015'), validEan13('2000000000016'), validEan13('20000')] };
        }""", [l for l in lines])
    assert got["valid"] == [True, False, False]
    for item, read in zip(lines, got["read"]):
        assert read == item["want"], (item["want"], read)


@needs_text
def test_safety_barcode_from_the_local_list_beside_a_printed_label():
    scene = sc.barcode_scene("2000000000015", right=((250, 250, 250), [("Glass Cleaner", 76, True), ("with AMMONIA", 64, True)],
                                                     (20, 60, 160)), name="assist_barcode_pair.png")
    with browser_page(f"/assist?mode=safety&hand=0&src={scene}", viewport=GLASSES) as (page, _):
        d = _wait_verdict(page)
        assert d["products"] == [{"code": "2000000000015", "name": "Lab bleach (in-house label)", "source": "local"}]
        names = sorted(i["name"] for i in d["items"])
        assert names == ["Glass cleaner (often contains ammonia)", "Lab bleach (in-house label)"], d["items"]
        assert d["verdict"] == {"level": "danger", "rules": ["bleach-ammonia"]}
        assert not page.problems, page.problems


def test_barcode_online_lookup_only_when_switched_on():
    code = sc.ean13_check("200000000003")                                 # not in the local list
    scene = sc.barcode_scene(code, name="assist_barcode_online.png", sideways=True)
    product = {"status": 1, "product": {"product_name": "Hazelnut Energy Drink", "categories": "Beverages, Energy drinks",
                                        "ingredients_text": "carbonated water, sugar, caffeine, hazelnut flavouring",
                                        "allergens_tags": ["en:nuts"]}}
    asked = []

    def off(route):
        asked.append(route.request.url)
        route.fulfill(status=200, content_type="application/json", headers={"Access-Control-Allow-Origin": "*"},
                      body=__import__("json").dumps(product))

    with browser_page("/static/blank.html", viewport=GLASSES) as (page, base):
        page.route("https://world.openfoodfacts.org/**", off)
        page.goto(base + f"/assist?mode=safety&hand=0&src={scene}")
        page.wait_for_function("() => window.assist?.state().detail?.scans >= 6 && document.getElementById('toast').textContent.includes('not in the local list')",
                               timeout=120000)
        assert asked == [], "nothing leaves the device unless Online lookup is on"
        assert detail(page)["items"] == []
        page.click("[data-act=online]")
        assert "Open Food Facts" in page.locator("#toast").inner_text()
        page.wait_for_function("() => window.assist.state().detail.items.length === 1", timeout=60000)
        d = detail(page)
        assert len(asked) == 1 and code in asked[0]
        item = d["items"][0]
        assert item["name"] == "Hazelnut Energy Drink" and "high-caffeine" in item["tags"] and item["allergens"] == ["tree nuts"]
        assert d["products"][0]["source"] == "online"
        # the choice is remembered, and the product is kept for next time without asking again
        assert page.evaluate("() => JSON.parse(localStorage.getItem('cvision-assist')).onlineLookup") is True
        assert code in page.evaluate("() => localStorage.getItem('cvision-assist-products')")
