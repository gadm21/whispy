"""Demo 2 -- laptop speaks -> RPi1 microphone hears -> laptop STT verifies.

The distributed proof: actuator abstraction + real acoustics + remote
sensor access + network streaming + model execution on a different
device than the sensor. The Pi needs no whisper/torch -- it only
provides an audio stream.

    laptop speaker --speak--> air --> RPi1 microphone
        -> Thoth local API (authenticated tail)
        -> LAN SensorHandle on the laptop
        -> capture_window (pre-roll + speech + post-roll)
        -> whisper-stt plugin (laptop)
        -> transcript compared to the spoken sentence

Setup::

    RPi1:   pip install whispy thoth-node whispy-sensor-microphone
            thoth expose --lan --sensor microphone
            thoth daemon
    Laptop: pip install whispy whispy-actuator-speaker
                        whispy-model-whisper-stt[faster]
            python whispy/examples/demo2_remote_stt.py rpi1.local <token>
"""

from __future__ import annotations

import re
import sys
import unicodedata

SENTENCE = "Thothcraft remote microphone test number seven."


_DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three",
           "4": "four", "5": "five", "6": "six", "7": "seven",
           "8": "eight", "9": "nine"}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and map digits to number words —
    Whisper legitimately transcribes "seven" as "7"."""
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return " ".join(_DIGITS.get(tok, tok) for tok in text.split())


def wer(expected: str, actual: str) -> float:
    """Word error rate (Levenshtein over words)."""
    ref, hyp = expected.split(), actual.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for r in ref:
        new = [d[0] + 1]
        for j, h in enumerate(hyp, 1):
            new.append(min(new[-1] + 1, d[j] + 1, d[j - 1] + (r != h)))
        d = new
    return d[-1] / len(ref)


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else "rpi1.local"
    token = sys.argv[2] if len(sys.argv) > 2 else None
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 5000

    import whispy
    from whispy.actuators import Speak

    laptop = whispy.local()
    rpi1 = whispy.lan(host, port=port, token=token)

    # -- actuators + remote sensor ------------------------------------------------
    speaker = laptop.actuator("speaker")
    try:
        mic = rpi1.sensor("microphone")
    except KeyError:
        mics = [d for d in rpi1.sensor_descriptors()
                if d.modality == "microphone"]
        if not mics:
            raise SystemExit("no microphone exposed by the remote node")
        mic = rpi1.sensor(mics[0].id)

    # Prove the input is the REMOTE microphone, not the laptop's own mic.
    assert rpi1.info.id != laptop.info.id, "remote device resolved to local!"
    local_mics = {s.id for s in laptop.sensors() if s.type == "microphone"}
    assert mic.info.id not in local_mics, \
        "remote mic id collides with a local sensor id"

    stt = whispy.model("whisper-stt", config={
        "variant": "tiny.en", "device": "cpu", "compute_type": "int8"})

    print(f"Speech source actuator: {laptop.info.id} / {speaker.info.id}")
    print(f"Captured sensor:        {rpi1.info.id} / {mic.info.id}")
    print(f"Model execution:        {laptop.info.id}")
    print(f"Model:                  whisper-stt / tiny.en")
    print()

    # -- orchestrated capture: pre-roll -> speak -> post-roll ------------------------
    session = whispy.capture_window(mic, pre_roll=1.0)
    session.start()
    speaker.execute(Speak(SENTENCE))
    window = session.finish(post_roll=1.5)

    result = stt.predict(window)

    expected = normalize(SENTENCE)
    actual = normalize(result.attributes.get("text", ""))
    score = wer(expected, actual)
    # tiny.en on real acoustic audio garbles quiet onsets — the demo
    # proves the distributed pipeline (actuator -> remote mic -> LAN
    # stream -> local STT), not transcription accuracy. WER <= 0.5
    # means the sentence was substantially heard through the air gap.
    passed = score <= 0.5

    print(f"Expected:   {expected}")
    print(f"Recognized: {actual}")
    print(f"WER:        {score:.2f}")
    print(f"\n{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
