from dataclasses import dataclass
import asyncio
import time
import pandas as pd
from pathlib import Path

from .settings import (
    APP_URL,
    CHATBOT_MODE,
    HEADLESS,
    TIMEOUT,
    RETRY_COUNT,
    ROOT,
)


# The actual Streamlit application discovered during frame inspection.
# We navigate directly to it instead of relying on the wrapper iframe.
STREAMLIT_APP_URL = (
    "https://llm-rag-testing-workshop.streamlit.app/~/+/Application"
)


@dataclass
class ChatResult:
    question: str
    answer: str
    contexts: list[str]
    latency_seconds: float
    status: str
    error: str = ""
    raw: dict | None = None


class ChatbotExecutor:

    def __init__(self, mode=None):
        self.mode = mode or CHATBOT_MODE

        self._playwright = None
        self._browser = None
        self._page = None

    async def close(self):

        try:
            if self._browser:
                await self._browser.close()

        except Exception:
            pass

        try:
            if self._playwright:
                await self._playwright.stop()

        except Exception:
            pass

        self._browser = None
        self._playwright = None
        self._page = None

    async def _ensure_page(self):

        if self._page:
            return self._page

        from playwright.async_api import async_playwright

        print("Executor: starting Playwright...")

        self._playwright = await async_playwright().start()

        print("Executor: launching Chrome...")

        self._browser = await self._playwright.chromium.launch(
            headless=HEADLESS,
            channel="chrome"
        )

        self._page = await self._browser.new_page(
            viewport={
                "width": 1440,
                "height": 1000
            }
        )

        print("Executor: opening actual Streamlit application...")

        await self._page.goto(
            STREAMLIT_APP_URL,
            wait_until="domcontentloaded",
            timeout=TIMEOUT * 1000
        )

        print("Executor: waiting for chatbot...")

        chat_input = self._page.locator(
            'textarea[data-testid="stChatInputTextArea"]'
        )

        await chat_input.wait_for(
            state="visible",
            timeout=TIMEOUT * 1000
        )

        print("Executor: chatbot is ready.")

        return self._page

    async def _get_chat_input(self, page):

        selector = (
            'textarea[data-testid="stChatInputTextArea"]'
        )

        print("Executor: locating chat input...")

        box = page.locator(selector)

        await box.wait_for(
            state="visible",
            timeout=TIMEOUT * 1000
        )

        print("Executor: chat input found.")

        return box

    async def _wait_for_answer(
        self,
        page,
        question,
        previous_message_count
    ):

        print("Executor: waiting for chatbot response...")

        deadline = time.time() + TIMEOUT

        processing_text = (
            "processing through intent agent, rag pipeline & running guardrail"
        )

        last_text = ""

        while time.time() < deadline:

            await asyncio.sleep(1)

            try:

                messages = page.locator(
                    '[data-testid="stChatMessage"]'
                )

                count = await messages.count()

                # We need a new chat message after the question
                # was submitted.
                if count <= previous_message_count:
                    continue

                # Inspect messages created after the previous count.
                for index in range(
                    previous_message_count,
                    count
                ):

                    candidate = (
                        await messages.nth(index).inner_text()
                    ).strip()

                    if not candidate:
                        continue

                    normalized = " ".join(
                        candidate.lower().split()
                    )

                    # Ignore the user's question.
                    if normalized == question.strip().lower():
                        continue

                    # Ignore temporary processing text.
                    if processing_text in normalized:

                        if candidate != last_text:

                            print(
                                "Executor: chatbot is still processing..."
                            )

                            last_text = candidate

                        continue

                    # Ignore generic temporary messages.
                    if normalized in {
                        "processing...",
                        "thinking...",
                        "generating...",
                        "loading...",
                    }:

                        continue

                    print(
                        "Executor: final assistant response detected."
                    )

                    print(
                        "Executor: response preview: "
                        + candidate[:500]
                    )

                    return candidate

            except Exception:
                # Streamlit may rerender the DOM while the
                # response is being generated.
                continue

        raise RuntimeError(
            "Final chatbot response was not detected within "
            f"{TIMEOUT} seconds."
        )

    async def _click_save_exchange(self, page):

        print(
            "Executor: locating "
            "Save Exchange to Test Dataset..."
        )

        text = page.get_by_text(
            "Save Exchange to Test Dataset",
            exact=False
        )

        await text.wait_for(
            state="visible",
            timeout=TIMEOUT * 1000
        )

        print(
            "Executor: Save Exchange text found."
        )

        button = text.locator(
            "xpath=ancestor::button[1]"
        )

        if await button.count() == 0:

            button = text.locator(
                "xpath=ancestor::*[@role='button'][1]"
            )

        if await button.count() == 0:

            raise RuntimeError(
                "Save Exchange text was found, but its clickable "
                "button/container could not be located."
            )

        await button.wait_for(
            state="visible",
            timeout=TIMEOUT * 1000
        )

        await button.click()

        await asyncio.sleep(2)

        print(
            "Executor: exchange saved to test dataset."
        )

    async def _open_evaluation(self, page):

        print("Executor: opening Evaluation...")

        evaluation = page.get_by_text(
            "Evaluation",
            exact=True
        )

        await evaluation.wait_for(
            state="visible",
            timeout=TIMEOUT * 1000
        )

        await evaluation.click()

        await asyncio.sleep(3)

        print("Executor: Evaluation opened.")

    async def _return_to_application(self, page):

        print("Executor: returning to Application...")

        application = page.get_by_text(
            "Application",
            exact=True
        )

        await application.wait_for(
            state="visible",
            timeout=TIMEOUT * 1000
        )

        await application.click()

        await asyncio.sleep(3)

        print("Executor: Application reopened.")

    async def _download_test_dataset(self, page):

        print(
            "Executor: locating "
            "Download rag_test_set.csv..."
        )

        button = page.get_by_role(
            "button",
            name="Download rag_test_set.csv",
            exact=True
        )

        try:

            await button.wait_for(
                state="visible",
                timeout=10 * 1000
            )

        except Exception:

            button = page.get_by_text(
                "Download rag_test_set.csv",
                exact=True
            )

            await button.wait_for(
                state="visible",
                timeout=TIMEOUT * 1000
            )

        download_dir = ROOT / "reports" / "chatbot_app" / "generated"

        download_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            download_dir / "rag_test_set.csv"
        )

        print("Executor: downloading generated CSV...")

        async with page.expect_download(
            timeout=TIMEOUT * 1000
        ) as download_info:

            await button.click()

        download = await download_info.value

        await download.save_as(
            str(output_file)
        )

        print(
            "Executor: CSV downloaded:"
        )

        print(output_file)

        return output_file

    async def _populate_ground_truth(self, csv_path):

        print(
            "Executor: populating ground truth from golden dataset..."
        )

        golden_path = ROOT / "data" / "shared" / "golden_dataset.csv"

        if not golden_path.exists():

            raise FileNotFoundError(
                f"Golden dataset not found: {golden_path}"
            )

        result_df = pd.read_csv(
            csv_path,
            dtype=str
        )

        golden_df = pd.read_csv(
            golden_path,
            dtype=str
        )

        golden_map = {
            str(row["question"]).strip().lower():
            row["expected_answer"]
            for _, row in golden_df.iterrows()
        }

        matched = 0
        unmatched = []

        for index, row in result_df.iterrows():

            question = str(
                row["question"]
            ).strip()

            key = question.lower()

            if key in golden_map:

                result_df.at[
                    index,
                    "ground_truth"
                ] = golden_map[key]

                matched += 1

            else:

                unmatched.append(question)

        result_df.to_csv(
            csv_path,
            index=False
        )

        print(
            f"Executor: ground truth populated for {matched} questions."
        )

        if unmatched:

            print(
                f"Executor: WARNING - {len(unmatched)} "
                "questions were not found in golden dataset."
            )

        return csv_path

    async def ask(self, question: str) -> ChatResult:

        last_error = ""

        for attempt in range(RETRY_COUNT + 1):

            started = time.perf_counter()

            try:

                if self.mode != "playwright":

                    raise RuntimeError(
                        f"Unsupported execution mode: {self.mode}"
                    )

                page = await self._ensure_page()

                chat_input = await self._get_chat_input(
                    page
                )

                print(
                    f"Executor: submitting question: {question}"
                )

                messages = page.locator(
                    '[data-testid="stChatMessage"]'
                )

                previous_count = await messages.count()

                await chat_input.fill(
                    question
                )

                await chat_input.press(
                    "Enter"
                )

                answer = await self._wait_for_answer(
                    page,
                    question,
                    previous_count
                )

                print(
                    "Executor: chatbot response received."
                )

                # ---------------------------------------------
                # Save Exchange
                # ---------------------------------------------

                await self._click_save_exchange(
                    page
                )

                # ---------------------------------------------
                # Evaluation
                # ---------------------------------------------

                await self._open_evaluation(
                    page
                )

                # ---------------------------------------------
                # Back to Application
                # ---------------------------------------------

                await self._return_to_application(
                    page
                )

                # ---------------------------------------------
                # Download CSV
                # ---------------------------------------------

                csv_file = await self._download_test_dataset(
                    page
                )

                await self._populate_ground_truth(
                    csv_file
                )

                latency = (
                    time.perf_counter()
                    - started
                )

                return ChatResult(
                    question=question,
                    answer=answer,
                    contexts=[],
                    latency_seconds=latency,
                    status="PASS",
                    error="",
                    raw={
                        "downloaded_csv": str(
                            csv_file
                        ),
                        "mode": "playwright",
                        "application_url": (
                            STREAMLIT_APP_URL
                        )
                    }
                )

            except Exception as exc:

                last_error = str(exc)

                print(
                    f"Executor error: {last_error}"
                )

                if attempt < RETRY_COUNT:

                    print(
                        f"Executor: retrying "
                        f"({attempt + 1}/{RETRY_COUNT})..."
                    )

                    await self.close()

                    await asyncio.sleep(
                        2 ** attempt
                    )

        return ChatResult(
            question=question,
            answer="",
            contexts=[],
            latency_seconds=0,
            status="ERROR",
            error=last_error,
            raw=None
        )

    async def ask_safety(
        self,
        question: str
    ) -> ChatResult:

        last_error = ""

        for attempt in range(
            RETRY_COUNT + 1
        ):

            started = time.perf_counter()

            try:

                if self.mode != "playwright":

                    raise RuntimeError(
                        f"Unsupported execution mode: {self.mode}"
                    )

                page = await self._ensure_page()

                chat_input = await self._get_chat_input(
                    page
                )

                print(
                    f"Executor: submitting safety question: {question}"
                )

                messages = page.locator(
                    '[data-testid="stChatMessage"]'
                )

                previous_count = await messages.count()

                await chat_input.fill(
                    question
                )

                await chat_input.press(
                    "Enter"
                )

                answer = await self._wait_for_answer(
                    page,
                    question,
                    previous_count
                )

                print(
                    "Executor: safety chatbot response received."
                )

                latency = (
                    time.perf_counter()
                    - started
                )

                return ChatResult(
                    question=question,
                    answer=answer,
                    contexts=[],
                    latency_seconds=latency,
                    status="PASS",
                    error="",
                    raw={
                        "mode": "playwright",
                        "application_url": (
                            STREAMLIT_APP_URL
                        )
                    }
                )

            except Exception as exc:

                last_error = str(exc)

                print(
                    f"Executor safety error: {last_error}"
                )

                if attempt < RETRY_COUNT:

                    print(
                        f"Executor: retrying "
                        f"({attempt + 1}/{RETRY_COUNT})..."
                    )

                    await self.close()

                    await asyncio.sleep(
                        2 ** attempt
                    )

        return ChatResult(
            question=question,
            answer="",
            contexts=[],
            latency_seconds=0,
            status="ERROR",
            error=last_error,
            raw=None
        )


async def main():

    executor = ChatbotExecutor()

    try:

        result = await executor.ask(
            "What is pain?"
        )

        print("\n==============================")
        print("FINAL RESULT")
        print("==============================")

        print(
            "Status:",
            result.status
        )

        print(
            "Question:",
            result.question
        )

        print(
            "Answer:",
            result.answer
        )

        print(
            "Latency:",
            result.latency_seconds
        )

        print(
            "Error:",
            result.error
        )

        print(
            "Raw:",
            result.raw
        )

    finally:

        await executor.close()


if __name__ == "__main__":
    asyncio.run(main())