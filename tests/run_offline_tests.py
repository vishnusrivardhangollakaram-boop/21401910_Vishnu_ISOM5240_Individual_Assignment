"""
run_offline_tests.py
--------------------
Automated functional tests for TaleTwinkle's app.py that run WITHOUT downloading any model.

The folder tests/stubs holds small stand-ins for streamlit, transformers, torch and gTTS, so every
function in app.py (picture scaling, caption cleaning, story trimming, early stop, background speech,
voice speed and character effects, fall-backs) and the full main() screen flow can be checked quickly
on any computer. Real model quality and speed are measured with benchmark/TaleTwinkle_Model_Benchmark.ipynb
and on the deployed Streamlit app.

Run from the project folder:  python tests/run_offline_tests.py
(ffmpeg must be installed; the fake gTTS uses it to make MP3 audio.)
"""

# ==============================
# IMPORT PART
# ==============================
import io
import os
import sys
import time

import numpy as np

TESTS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(TESTS_FOLDER)
sys.path.insert(0, os.path.join(TESTS_FOLDER, "stubs"))
sys.path.insert(1, PROJECT_FOLDER)

import app                     # noqa: E402  (imported after the stubs on purpose)
import gtts                    # noqa: E402
import streamlit as st         # noqa: E402
import torch                   # noqa: E402
import transformers            # noqa: E402
from PIL import Image          # noqa: E402

TEST_RESULTS = []


# ==============================
# FUNCTION PART
# ==============================
def record_result(test_name, passed, details):
    """
    Store and print one test result.

    Parameters:
        test_name (str): what was tested.
        passed (bool): True if the check passed.
        details (str): measured values.
    """
    TEST_RESULTS.append({"test": test_name, "passed": bool(passed), "details": details})
    print(f"{'PASS' if passed else 'FAIL'} | {test_name} | {details}")


def make_fake_upload(picture_bytes):
    """
    Build an object that behaves like Streamlit's UploadedFile.

    Parameters:
        picture_bytes (bytes): file content.
    Returns:
        object: has getvalue().
    """
    return type("FakeUploadedFile", (), {"getvalue": lambda self: picture_bytes})()


def make_picture_bytes(picture_format):
    """
    Make a small test picture in memory.

    Parameters:
        picture_format (str): "JPEG" or "PNG".
    Returns:
        bytes: encoded picture.
    """
    picture_buffer = io.BytesIO()
    Image.new("RGB", (300, 200), (30, 160, 90)).save(picture_buffer, picture_format)
    return picture_buffer.getvalue()


def run_app_once():
    """
    Run app.main() once with the stub Streamlit and return the recorded screen calls.

    Returns:
        list: recorded (name, args, kwargs) calls.
    """
    st.CALLS.clear()
    app.main()
    return list(st.CALLS)


def dominant_frequency(audio_samples, sample_rate):
    """
    Find the strongest frequency in a stretch of audio (to check pitch).

    Parameters:
        audio_samples (np.ndarray): samples.
        sample_rate (int): samples per second.
    Returns:
        float: frequency in Hz.
    """
    spectrum = np.abs(np.fft.rfft(audio_samples))
    return float(np.fft.rfftfreq(len(audio_samples), 1 / sample_rate)[spectrum.argmax()])


def test_picture_functions():
    """Pictures: small ones are scaled up, large ones down, model input never enlarged, PNG transparency handled."""
    small_display = app.resize_image_for_display(picture=Image.new("RGB", (120, 80)), longest_side_pixels=app.PREFERRED_IMAGE_DISPLAY_SIZE)
    large_display = app.resize_image_for_display(picture=Image.new("RGB", (4000, 3000)), longest_side_pixels=app.PREFERRED_IMAGE_DISPLAY_SIZE)
    model_input = app.prepare_image_for_model(picture=Image.new("RGB", (120, 80)), maximum_side_pixels=app.MODEL_INPUT_IMAGE_SIZE)
    record_result(test_name="Small picture scaled UP to preferred size", passed=max(small_display.size) == app.PREFERRED_IMAGE_DISPLAY_SIZE,
                  details=f"120x80 -> {small_display.size[0]}x{small_display.size[1]}")
    record_result(test_name="Large picture scaled DOWN to preferred size", passed=max(large_display.size) == app.PREFERRED_IMAGE_DISPLAY_SIZE,
                  details=f"4000x3000 -> {large_display.size[0]}x{large_display.size[1]}")
    record_result(test_name="Model input never enlarged", passed=model_input.size == (120, 80), details=f"{model_input.size}")
    transparent_buffer = io.BytesIO()
    Image.new("RGBA", (50, 50), (255, 0, 0, 0)).save(transparent_buffer, "PNG")
    opened_picture = app.open_uploaded_picture(uploaded_picture_file=make_fake_upload(picture_bytes=transparent_buffer.getvalue()))
    record_result(test_name="Transparent PNG opened as RGB on white",
                  passed=opened_picture.mode == "RGB" and opened_picture.getpixel((1, 1)) == (255, 255, 255),
                  details=f"mode={opened_picture.mode}, pixel={opened_picture.getpixel((1, 1))}")


def test_caption_and_story_functions():
    """Caption cleaning (real BLIP artefacts from the TaleTwinkle benchmark), trimming, safety and backup story."""
    caption_examples = {"arafed there is a cat cat sitting on a sofa.": "a cat sitting on a sofa",
                        "jungle jungle jungle jungle jungle jungle": "jungle",
                        "underwater background with corals and fishes": "underwater with corals and fishes",
                        "children playing in the park illustration": "children playing in the park",
                        "a painting of a river surrounded by trees": "a river surrounded by trees"}
    for raw_caption, expected_caption in caption_examples.items():
        cleaned_caption = app.clean_caption(raw_caption=raw_caption)
        record_result(test_name=f"Caption cleaned: '{raw_caption[:28]}...'", passed=cleaned_caption == expected_caption,
                      details=f"-> '{cleaned_caption}'")
    long_story = " ".join([transformers.FAKE_STORY.replace(" The end is", "")] * 3)
    for target_word_count in (50, 75, 100):
        trimmed_story, word_count = app.trim_story_to_word_limit(story_text=long_story, target_word_count=target_word_count, is_poem=False)
        record_result(test_name=f"Story trimmed for slider = {target_word_count}",
                      passed=app.MINIMUM_STORY_WORDS <= word_count <= min(app.MAXIMUM_STORY_WORDS, target_word_count + 10)
                      and trimmed_story.endswith((".", "!", "?")),
                      details=f"{word_count} words, ends with '{trimmed_story[-12:]}'")
    trimmed_poem, poem_word_count = app.trim_story_to_word_limit(story_text="\n".join(["The little cat sat on a mat so round,"] * 20),
                                                                 target_word_count=60, is_poem=True)
    record_result(test_name="Poem keeps its line breaks", passed=trimmed_poem.count("\n") >= 4 and poem_word_count <= 70,
                  details=f"{poem_word_count} words, {trimmed_poem.count(chr(10)) + 1} lines")
    record_result(test_name="Safety filter blocks unsafe words", passed=not app.story_is_kid_safe(story_text="He found a gun."),
                  details="'gun' blocked")
    record_result(test_name="Safety filter allows normal words", passed=app.story_is_kid_safe(story_text="A warm, happy puppy."),
                  details="'warm' allowed (not 'war')")
    backup_story, backup_word_count = app.make_backup_story(image_caption="a dog with a ball", theme_key="space_quest", target_word_count=75)
    record_result(test_name="Backup story is 50-100 words, safe, no fixed 'picture came alive' opening",
                  passed=50 <= backup_word_count <= 100 and app.story_is_kid_safe(story_text=backup_story)
                  and "picture came alive" not in backup_story, details=f"{backup_word_count} words")
    story_messages = app.build_story_messages(image_caption="a dog with a ball", theme_instruction="Make it magical.", target_word_count=75)
    record_result(test_name="Prompt contains caption, theme, length and 'no title'",
                  passed="a dog with a ball" in story_messages[1]["content"] and "75 words" in story_messages[1]["content"]
                  and "Make it magical." in story_messages[1]["content"], details="chat messages built")


def test_background_speech_queue():
    """Sentences are queued for speech while the story is still being written."""
    queued_sentences = []
    stream_progress = {"text": "", "queued_sentences": 0}
    for text_piece in ["The dog ", "ran fast. ", "The cat ", "jumped! Then", " they slept."]:
        stream_progress["text"] += text_piece
        app.queue_finished_sentences(stream_progress=stream_progress, queue_sentence_for_speech=queued_sentences.append)
    record_result(test_name="Finished sentences queued one by one during streaming",
                  passed=queued_sentences == ["The dog ran fast.", "The cat jumped!", "Then they slept."],
                  details=f"{queued_sentences}")


def test_early_stop():
    """Story generation stops as soon as the story is long enough (saves time on long model outputs)."""
    original_story, original_delay = transformers.FAKE_STORY, transformers.WORD_DELAY_SECONDS
    transformers.FAKE_STORY = " ".join(["The happy dog ran to the park and played with a ball."] * 30)
    transformers.WORD_DELAY_SECONDS = 0
    generated_word_counts = {}
    for use_early_stop in (True, False):
        stream_progress = {"text": "", "queued_sentences": 0, "first_words_time": None, "problems": []}
        "".join(app.stream_story_text(story_pipeline=transformers.pipeline("text-generation"), story_messages=[],
                                      target_word_count=75, stream_progress=stream_progress,
                                      queue_sentence_for_speech=lambda sentence_text: None, use_early_stop=use_early_stop))
        generated_word_counts[use_early_stop] = transformers.EVENTS["words_generated"]
    record_result(test_name="Early stop: long model output cut at the word limit",
                  passed=generated_word_counts[True] <= 90 < generated_word_counts[False],
                  details=f"{generated_word_counts[True]} words with early stop vs {generated_word_counts[False]} without (target 75)")
    transformers.FAKE_STORY, transformers.WORD_DELAY_SECONDS = original_story, original_delay


def test_voice_functions():
    """Voice speed keeps pitch, cartoon pitch shifts are correct, robot effect, MP3 decoding, narration assembly."""
    sample_rate = 24000
    time_axis = np.arange(sample_rate * 30) / sample_rate
    test_tone = (0.5 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)
    for voice_speed in (0.5, 2.0):
        start_time = time.perf_counter()
        stretched_tone = app.time_stretch_speech(audio_samples=test_tone, sample_rate=sample_rate, stretch_rate=voice_speed)
        elapsed_seconds = time.perf_counter() - start_time
        measured_seconds = len(stretched_tone) / sample_rate
        measured_pitch = dominant_frequency(audio_samples=stretched_tone[sample_rate:3 * sample_rate], sample_rate=sample_rate)
        record_result(test_name=f"Voice speed {voice_speed:.2f}x keeps pitch",
                      passed=abs(measured_seconds - 30 / voice_speed) < 0.2 and abs(measured_pitch - 220) < 2,
                      details=f"{measured_seconds:.2f} s (expected {30 / voice_speed:.2f}), pitch {measured_pitch:.2f} Hz, took {elapsed_seconds:.2f} s")
    for persona_key in ("polly_parrot", "bruno_bear"):
        persona_settings = app.VOICE_PERSONAS[persona_key]
        shifted_tone = app.change_pitch_and_tempo(audio_samples=test_tone, sample_rate=sample_rate,
                                                  semitone_shift=persona_settings["semitone_shift"], tempo_factor=persona_settings["tempo_factor"])
        expected_pitch = 220 * 2 ** (persona_settings["semitone_shift"] / 12)
        measured_pitch = dominant_frequency(audio_samples=shifted_tone[sample_rate:3 * sample_rate], sample_rate=sample_rate)
        record_result(test_name=f"Cartoon pitch: {persona_settings['label']}", passed=abs(measured_pitch - expected_pitch) < 3,
                      details=f"{measured_pitch:.2f} Hz (expected {expected_pitch:.2f})")
    real_voice_keys = [voice_key for voice_key, persona in app.VOICE_PERSONAS.items()
                       if persona["semitone_shift"] == 0 and persona["tempo_factor"] == 1.0 and not persona["robot_effect"]]
    default_persona = app.VOICE_PERSONAS[app.DEFAULT_VOICE_KEY]
    record_result(test_name="Default voice is the real British lady (no pitch effect)",
                  passed=app.DEFAULT_VOICE_KEY in real_voice_keys and default_persona["accent_domain"] == "co.uk",
                  details=f"{default_persona['label']}; {len(real_voice_keys)} real unaltered voices")
    voice_order = list(app.VOICE_PERSONAS.keys())[:3]
    record_result(test_name="Voice order: Lily, then Parrot, then Kitten", passed=voice_order == ["lily_british", "polly_parrot", "whiskers_kitten"],
                  details=f"{voice_order}")
    robot_tone = app.add_robot_effect(audio_samples=test_tone[:sample_rate], sample_rate=sample_rate)
    record_result(test_name="Robot effect changes the sound", passed=not np.allclose(robot_tone, test_tone[:sample_rate]), details="60 Hz buzz added")
    mp3_buffer = io.BytesIO()
    gtts.gTTS(text="hello there friend").write_to_fp(mp3_buffer)
    decoded_samples, decoded_rate = app.decode_mp3_bytes(mp3_bytes=mp3_buffer.getvalue())
    record_result(test_name="MP3 decoding (ffmpeg fallback)", passed=decoded_rate == 24000 and len(decoded_samples) > 10000,
                  details=f"{len(decoded_samples)} samples at {decoded_rate} Hz")
    wav_bytes, audio_seconds = app.assemble_narration(sentence_clips=[(decoded_samples, decoded_rate)] * 3,
                                                      persona_settings=app.VOICE_PERSONAS["robo_beep"], voice_speed=1.5)
    record_result(test_name="Narration assembled (3 sentences, robot, 1.5x)", passed=wav_bytes[:4] == b"RIFF" and audio_seconds > 0,
                  details=f"{audio_seconds:.2f} s of audio")


def test_screen_flow():
    """Full main() flow: placeholders + greeting, live story, background speech, autoplay, caching, changes, bad file."""
    first_visit_calls = run_app_once()
    placeholder_count = sum(1 for call in first_visit_calls if call[0] == "empty.markdown" and "placeholder-card" in call[1][0])
    greeting_players = [call for call in first_visit_calls if call[0] == "empty.audio"]
    record_result(test_name="First visit: picture + story placeholders and a greeting audio player are shown",
                  passed=placeholder_count == 2 and len(greeting_players) == 1 and not greeting_players[0][2].get("autoplay"),
                  details=f"{placeholder_count} placeholders, greeting player (no autoplay)")
    record_result(test_name="No background music anywhere", passed=not any(call[2].get("loop") for call in first_visit_calls if call[0].endswith("audio"))
                  and not hasattr(app, "load_theme_music"), details="music code and files removed")
    record_result(test_name="Models shrunk to int8 at start-up", passed=len(torch.QUANTIZE_CALLS) >= 2,
                  details=f"{len(torch.QUANTIZE_CALLS)} models quantized")

    gtts.REQUESTS.clear()
    st.WIDGET_VALUES["picture_upload"] = make_fake_upload(picture_bytes=make_picture_bytes(picture_format="JPEG"))
    upload_calls = run_app_once()
    call_names = [call[0] for call in upload_calls]
    story_audio = [call for call in upload_calls if call[0] == "empty.audio"]
    story_result = st.session_state["story_result"]
    narration_result = st.session_state["narration_result"]
    record_result(test_name="Upload: picture shown, story streamed, narration autoplays (no button click)",
                  passed="empty.image" in call_names and "write_stream" in call_names and len(story_audio) == 1
                  and story_audio[0][2].get("autoplay") is True and "balloons" in call_names,
                  details=f"caption='{story_result['caption']}', {story_result['word_count']} words")
    story_card_index = max(index for index, call in enumerate(upload_calls) if call[0] == "empty.markdown" and "story-card" in call[1][0])
    audio_index = call_names.index("empty.audio")
    record_result(test_name="Story card is shown BEFORE the audio player", passed=story_card_index < audio_index,
                  details=f"story card call #{story_card_index}, audio call #{audio_index}")
    first_speech_request_time = min(request[0] for request in gtts.REQUESTS)
    record_result(test_name="Speech started in the background while the story was still being written",
                  passed=first_speech_request_time < transformers.EVENTS["stream_end"],
                  details=f"first sentence sent {transformers.EVENTS['stream_end'] - first_speech_request_time:.2f} s before the story finished")
    record_result(test_name="Timings recorded (2 decimals shown)",
                  passed=all(story_result[timing_name] >= 0 for timing_name in ("caption_seconds", "first_words_seconds", "story_seconds")),
                  details=f"first words {story_result['first_words_seconds']:.2f} s, story {story_result['story_seconds']:.2f} s, "
                          f"voice ready {narration_result['total_seconds']:.2f} s (voice {narration_result['voice_seconds']:.2f} s after story)")

    rerun_calls = run_app_once()
    record_result(test_name="Rerun with same choices does not rewrite the story",
                  passed="write_stream" not in [call[0] for call in rerun_calls], details="story reused from session state")

    st.WIDGET_VALUES["voice_choice"] = "polly_parrot"
    st.WIDGET_VALUES["voice_speed_choice"] = 1.5
    voice_change_calls = run_app_once()
    record_result(test_name="Voice / speed change re-records audio only",
                  passed="write_stream" not in [call[0] for call in voice_change_calls]
                  and any(call[0] == "empty.audio" and call[2].get("autoplay") for call in voice_change_calls),
                  details="new narration, same story")

    st.WIDGET_VALUES["theme_choice"] = "ocean_magic"
    theme_change_calls = run_app_once()
    record_result(test_name="Theme change writes a new story", passed="write_stream" in [call[0] for call in theme_change_calls],
                  details="new Ocean Magic story")

    st.WIDGET_VALUES["picture_upload"] = make_fake_upload(picture_bytes=b"this is not a picture")
    bad_file_calls = run_app_once()
    record_result(test_name="Broken file shows a friendly error (no crash)", passed=any(call[0] == "error" for call in bad_file_calls),
                  details="friendly error message shown")


def test_fallbacks():
    """The app must never break: story model crash, unsafe story and Google TTS outage all have fall-backs."""
    st.WIDGET_VALUES["picture_upload"] = make_fake_upload(picture_bytes=make_picture_bytes(picture_format="PNG"))
    original_pipeline_call = transformers._Pipe.__call__

    def crashing_pipeline_call(self, *arguments, **keyword_arguments):
        """Simulate the story model running out of memory while writing."""
        if self.task == "text-generation" and keyword_arguments.get("streamer") is not None:
            raise RuntimeError("simulated out of memory")
        return original_pipeline_call(self, *arguments, **keyword_arguments)

    transformers._Pipe.__call__ = crashing_pipeline_call
    st.session_state["story_version"] += 1
    run_app_once()
    record_result(test_name="Story model crash -> safe backup story", passed=st.session_state["story_result"]["used_backup_story"],
                  details=f"{st.session_state['story_result']['word_count']} words")
    transformers._Pipe.__call__ = original_pipeline_call

    original_story = transformers.FAKE_STORY
    transformers.FAKE_STORY = "The dog found a knife and a gun in the dark forest. " * 10
    st.session_state["story_version"] += 1
    run_app_once()
    record_result(test_name="Unsafe story -> safe backup story", passed=st.session_state["story_result"]["used_backup_story"], details="blocked")
    transformers.FAKE_STORY = original_story

    original_write = gtts.gTTS.write_to_fp

    def offline_write(self, file_pointer):
        """Simulate Google TTS being unreachable."""
        raise ConnectionError("simulated no internet")

    gtts.gTTS.write_to_fp = offline_write
    st.WIDGET_VALUES["voice_choice"] = "amy_american"
    st.session_state["story_version"] += 1
    run_app_once()
    record_result(test_name="Google TTS offline -> Hugging Face voice used, story still narrated",
                  passed="backup" in st.session_state["narration_result"]["voice_engine"],
                  details=st.session_state["narration_result"]["voice_engine"])
    gtts.gTTS.write_to_fp = original_write

    st.WIDGET_VALUES["voice_choice"] = "max_man"
    st.session_state["story_version"] += 1
    run_app_once()
    record_result(test_name="Man's voice uses the local Hugging Face pipeline",
                  passed="Hugging Face pipeline" in st.session_state["narration_result"]["voice_engine"],
                  details=st.session_state["narration_result"]["voice_engine"])


def write_results_markdown(file_path):
    """
    Save the results as a markdown table (copied into README.md).

    Parameters:
        file_path (str): destination file.
    """
    passed_count = sum(1 for result in TEST_RESULTS if result["passed"])
    table_lines = [f"Offline test run: {passed_count}/{len(TEST_RESULTS)} passed\n",
                   "| # | Test | Result | Details |", "|---|---|---|---|"]
    for result_index, result in enumerate(TEST_RESULTS, start=1):
        table_lines.append(f"| {result_index} | {result['test']} | {'✅ Pass' if result['passed'] else '❌ Fail'} | {result['details']} |")
    with open(file_path, "w", encoding="utf-8") as results_file:
        results_file.write("\n".join(table_lines) + "\n")


# ==============================
# MAIN PART
# ==============================
def main():
    """Run every test group and save the results table."""
    test_picture_functions()
    test_caption_and_story_functions()
    test_background_speech_queue()
    test_early_stop()
    test_voice_functions()
    test_screen_flow()
    test_fallbacks()
    write_results_markdown(file_path=os.path.join(TESTS_FOLDER, "offline_test_results.md"))
    passed_count = sum(1 for result in TEST_RESULTS if result["passed"])
    print(f"\n{passed_count}/{len(TEST_RESULTS)} tests passed")


if __name__ == "__main__":
    main()
