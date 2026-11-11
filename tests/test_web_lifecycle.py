"""Camera lifecycle, optional model loading, and explicit server fallback."""
from tests.webharness import browser_page


def ready(page):
    page.wait_for_function("() => window.chemvision && !document.getElementById('go').disabled")


def test_camera_pause_resume_and_lazy_objects():
    with browser_page("/ar") as (page, _):
        ready(page)
        assert page.evaluate("() => window.chemvision.objects === null")
        assert not page.evaluate("() => performance.getEntriesByType('resource').some(r => r.name.endsWith('/objects.onnx'))")
        page.click("#go")
        page.wait_for_function("() => window.chemvision.running && window.chemvision.geo")
        page.evaluate("() => { window.oldTrack = document.getElementById('cam').srcObject.getVideoTracks()[0]; }")
        page.click("#pause")
        assert page.evaluate("() => oldTrack.readyState === 'ended' && !window.chemvision.running")
        assert page.locator("#start").is_visible()
        page.click("#go")
        page.wait_for_function("() => window.chemvision.running")
        assert page.evaluate("() => document.getElementById('cam').srcObject.getVideoTracks()[0] !== oldTrack")
        # Object weights are loaded only after their control is selected.
        if page.locator("#tObjects").is_visible():
            page.click("#tObjects")
            page.wait_for_function("() => !!window.chemvision.objects", timeout=60000)
            page.click("#tObjects")
            assert page.evaluate("() => !window.chemvision.on.objects && !window.chemvision.objectList.length")
        assert not page.problems, page.problems


def test_fallback_does_not_send_crops_without_opt_in():
    from pathlib import Path
    from tests.webharness import ROOT
    source = Path(ROOT) / "tests" / "fixtures" / "handwriting_cards.png"
    target = Path(ROOT) / "backend" / "static" / "test" / source.name
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(source.read_bytes())
    with browser_page() as (page, base):
        # Failure occurs before importing/creating a recognizer session.
        page.route("**/static/models/formula_rec.json", lambda route: route.fulfill(status=503, body="unavailable"))
        uploads = []
        page.on("request", lambda request: uploads.append(request.url) if request.method == "POST" else None)
        # A same-origin clip uses the video path and waits for Start camera.
        # For a still image, src autostarts, so stop at the start screen first.
        page.goto(base + f"/ar?src=/static/test/{source.name}")
        ready(page)
        page.wait_for_function("() => window.chemvision.tracker.tracks.length === 2")
        assert not page.locator("#serverFallback").is_checked()
        assert page.evaluate("() => window.chemvision.stats.mode") == "no reader"
        assert uploads == []
        page.click("#pause")
        assert page.locator("#fallback").is_visible()
        page.check("#serverFallback")
        assert page.evaluate("() => window.chemvision.stats.mode") == "server"
        page.click("#go")
        page.wait_for_function("() => window.chemvision.stats.reads > 0")
        assert any(url.endswith("/recognize") for url in uploads)


def test_bad_options_cannot_break_camera_layout():
    with browser_page("/ar?edge=Infinity&conf=not-a-number&objedge=-10") as (page, _):
        ready(page)
        page.click("#go")
        page.wait_for_function("() => window.chemvision.geo && window.chemvision.running")
        assert page.evaluate("() => Math.max(window.chemvision.geo.pw, window.chemvision.geo.ph)") == 448
        assert not page.problems, page.problems
