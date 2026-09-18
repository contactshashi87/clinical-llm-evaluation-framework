from playwright.sync_api import sync_playwright

URL = "https://llm-rag-testing-workshop.streamlit.app/Application"

with sync_playwright() as p:

    print("Opening application...")

    browser = p.chromium.launch(
        headless=False,
        channel="chrome"
    )

    page = browser.new_page()

    print("Going to page...")

    page.goto(
        URL,
        wait_until="domcontentloaded",
        timeout=120000
    )

    print("Waiting for iframe...")
    page.wait_for_timeout(15000)

    print("\nIFRAMES FOUND:")

    for i, frame in enumerate(page.frames):
        print(i, frame.url)

    # Find the actual Streamlit application iframe
    app_frame = None

    for frame in page.frames:
        if "/~/+/" in frame.url:
            app_frame = frame
            break

    if not app_frame:
        print("\nERROR: Application iframe not found.")
        input("Press ENTER to close...")
        browser.close()
        raise SystemExit

    print("\nAPPLICATION FRAME FOUND:")
    print(app_frame.url)

    print("\nWaiting for application to render...")
    app_frame.wait_for_timeout(15000)

    print("\nFRAME TITLE:")
    print(app_frame.title())

    print("\nFRAME BODY TEXT:")
    print(repr(app_frame.locator("body").inner_text()))

    print("\nTEXTAREAS:")
    print(app_frame.locator("textarea").count())

    for i in range(app_frame.locator("textarea").count()):
        element = app_frame.locator("textarea").nth(i)
        print(
            i,
            "placeholder=",
            element.get_attribute("placeholder"),
            "aria-label=",
            element.get_attribute("aria-label"),
            "data-testid=",
            element.get_attribute("data-testid")
        )

    print("\nINPUTS:")
    print(app_frame.locator("input").count())

    print("\nBUTTONS:")
    print(app_frame.locator("button").count())

    print("\nSTREAMLIT CHAT INPUT:")
    print(
        app_frame.locator("[data-testid='stChatInput']").count()
    )

    print("\nSTREAMLIT CHAT TEXTAREA:")
    print(
        app_frame.locator("[data-testid='stChatInputTextArea']").count()
    )

    page.screenshot(
        path="streamlit_iframe_debug.png",
        full_page=True
    )

    print("\nScreenshot saved: streamlit_iframe_debug.png")

    input("\nPress ENTER to close the browser...")

    browser.close()