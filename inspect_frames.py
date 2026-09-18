import asyncio
from playwright.async_api import async_playwright


async def main():

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=False,
            channel="chrome"
        )

        page = await browser.new_page(
            viewport={"width": 1440, "height": 1000}
        )

        url = "https://llm-rag-testing-workshop.streamlit.app/Application"

        print("Opening:", url)

        await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=120000
        )

        print("\nWaiting 30 seconds for application...")
        await page.wait_for_timeout(30000)

        print("\nPAGE URL:")
        print(page.url)

        print("\nFRAMES:")

        for i, frame in enumerate(page.frames):
            print(f"\nFRAME {i}")
            print("URL:", frame.url)

            try:
                title = await frame.title()
                print("TITLE:", title)
            except Exception as e:
                print("TITLE ERROR:", e)

            try:
                count = await frame.locator(
                    'textarea[data-testid="stChatInputTextArea"]'
                ).count()

                print("CHAT INPUT COUNT:", count)

            except Exception as e:
                print("CHAT INPUT ERROR:", e)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())