from pathlib import Path

from playwright.sync_api import sync_playwright

from src.visual_defects.mutations import apply_mutation_sync, revert_mutation_sync


def test_mutation_is_reversed():
    url = (Path(__file__).parents[1] / "fixtures/runtime_fixture.html").resolve().as_uri()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 800, "height": 600})
        page.goto(url)
        before = page.locator("#submit").bounding_box()
        apply_mutation_sync(page, "misalignment", "#submit", "#email")
        assert page.locator("#submit").bounding_box() != before
        result = revert_mutation_sync(page)
        assert result["reverted"] and page.locator("#submit").bounding_box() == before
        browser.close()
