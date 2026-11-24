"""Symptom survey: the questions and their scoring.

Seven sections of yes/no questions. Each main question scores 2 points when answered
"yes", and its follow-up items (only asked after a "yes") score 1 point each. A section's
level comes from its points as a share of ``max_score``: >= 50% high, >= 25% medium,
otherwise low. ``max_score`` values are the project's original scaling, slightly above
the reachable maximum on purpose. The overall ``symptom_score`` (0-100, higher = more
symptoms) is the points as a share of the summed ``max_score`` values.

This is a screening aid, not a diagnosis.
"""

from dataclasses import dataclass

MAIN_POINTS = 2
ITEM_POINTS = 1
HIGH_SHARE = 0.5
MEDIUM_SHARE = 0.25

LEVEL_LABELS = {
    "low": ("Few reported symptoms", "أعراض قليلة"),
    "medium": ("Some reported symptoms", "بعض الأعراض"),
    "high": ("Many reported symptoms", "أعراض كثيرة"),
}
NOTE_EN = "This survey is a screening aid, not a diagnosis. Please consult a physician for any concerns."
NOTE_AR = "هذا الاستبيان أداة فحص أولي وليس تشخيصاً. يُرجى استشارة الطبيب عند وجود أي قلق."


class SurveyInputError(ValueError):
    """Raised for invalid client input (maps to HTTP 400)."""


@dataclass(frozen=True)
class Item:
    id: str
    text_en: str
    text_ar: str


@dataclass(frozen=True)
class Question:
    id: str
    text_en: str
    text_ar: str
    follow_ups: tuple[Item, ...]


@dataclass(frozen=True)
class Section:
    key: str
    title_en: str
    title_ar: str
    max_score: int
    questions: tuple[Question, ...]


# Generated from the original frontend survey (texts and layout unchanged).
SECTIONS: tuple[Section, ...] = (
    Section(
        key='voice',
        title_en='Section 1: Voice and Speech',
        title_ar='القسم 1: الصوت والنُطق',
        max_score=12,
        questions=(
            Question('q1', 'Have you recently noticed a change in your voice?', 'هل لاحظت مؤخرًا إن صوتك اتغيّر؟', (
                Item('q1a', 'Has your voice become quieter or noticeably hoarse?', 'هل بقى صوتك أهدى أو فيه بُحّة ملحوظة؟'),
                Item('q1b', 'Do you feel you need to exert effort to speak clearly?', 'هل بتحس إنك بتحتاج تبذل مجهود علشان تتكلم بصوت واضح؟'),
                Item('q1c', "Are there moments when your pronunciation changes or isn't as precise as before?", 'هل في لحظات بتحس إن نطق الحروف بيتغير أو مش مضبوط زي قبل؟'),
                Item('q1d', 'Does your voice change depending on the time of day (worse in morning or night)?', 'هل صوتك بيتغير حسب وقت اليوم (أسوأ الصبح أو بالليل)؟'),
            )),
            Question('q2', 'Have you noticed that your voice tone or pitch shakes or changes abnormally?', 'هل لاحظت إن نغمة صوتك أو طبقة الصوت بتتهزّ أو بتتغير بشكل غير طبيعي؟', (
                Item('q2a', "Is this change constant or does it only appear when you're tired?", 'هل التغيّر ده مستمر ولا بيظهر بس لما تتعب؟'),
                Item('q2b', "Does it also happen when you're stressed or exerting effort?", 'هل بيحصل كمان لما تتوتر أو تبذل مجهود؟'),
            )),
        ),
    ),
    Section(
        key='movement',
        title_en='Section 2: Movement and Sensation',
        title_ar='القسم 2: الحركة والإحساس',
        max_score=15,
        questions=(
            Question('q3', 'Do you experience tingling or numbness in your hands or feet?', 'هل بيجيلك وخز أو تنميل في إيدك أو رجلك؟', (
                Item('q3a', 'Does the numbness last more than 24 hours?', 'هل التنميل بيستمر أكتر من 24 ساعة؟'),
                Item('q3b', 'Does it recur in the same place or move around?', 'هل بيتكرر في نفس المكان أو بيتنقل؟'),
                Item('q3c', 'Does it increase with heat or fatigue?', 'هل بيزيد مع الحرارة أو التعب؟'),
            )),
            Question('q4', 'Do you feel sudden or temporary weakness in your arm or leg?', 'هل بتحس بضعف مفاجئ أو مؤقت في ذراعك أو رجلك؟', (
                Item('q4a', 'Does the weakness temporarily prevent you from performing simple tasks?', 'هل الضعف بيمنعك مؤقتًا من أداء مهام بسيطة؟'),
                Item('q4b', 'Do you feel that muscle sensation returns after a while?', 'هل بتحس إن الإحساس بالعضلة بيرجع بعد وقت؟'),
            )),
            Question('q5', 'Have you suddenly lost your balance or felt the ground spinning?', 'هل حصل إنك فقدت توازنك فجأة أو حسيت إن الأرض بتلف؟', (
                Item('q5a', 'Do these episodes occur only while walking or standing?', 'هل بتحصل النوبات دي أثناء المشي أو الوقوف فقط؟'),
                Item('q5b', 'Do you feel you need to hold onto something to avoid falling?', 'هل بتحس إنك لازم تمسك حاجة علشان ما تقعش؟'),
            )),
        ),
    ),
    Section(
        key='vision',
        title_en='Section 3: Vision',
        title_ar='القسم 3: الرؤية',
        max_score=12,
        questions=(
            Question('q6', 'Has your vision suddenly weakened or have you seen fog in front of your eyes?', 'هل حصل إن نظرك ضعف فجأة أو شُفت ضباب قدام عينك؟', (
                Item('q6a', 'Is it in one eye only?', 'هل في عين واحدة فقط؟'),
                Item('q6b', 'Do you feel pain in the eye when moving?', 'هل بتحس بألم في العين وقت الحركة؟'),
                Item('q6c', 'Has your vision returned to normal after a few days or is it still affected?', 'هل رجع نظرك طبيعي بعد أيام أو لسه متأثر؟'),
            )),
            Question('q7', 'Do you see flashes or strange light spots sometimes?', 'هل بتشوف ومضات أو بقع ضوء غريبة أحيانًا؟', (
                Item('q7a', 'Does it happen with eye strain?', 'هل بتحصل مع إجهاد العين؟'),
                Item('q7b', 'Does it occur in the same eye or move around?', 'هل بتيجي في نفس العين أو بتتنقل؟'),
            )),
        ),
    ),
    Section(
        key='cognitive',
        title_en='Section 4: Focus and Energy',
        title_ar='القسم 4: التركيز والطاقة',
        max_score=10,
        questions=(
            Question('q8', 'Do you forget simple things more than before?', 'هل بتنسى حاجات بسيطة أكتر من الأول؟', (
                Item('q8a', 'Does forgetfulness affect work or study?', 'هل النسيان بيأثر على الشغل أو الدراسة؟'),
                Item('q8b', 'Does it increase with fatigue or lack of sleep?', 'هل بيزيد مع التعب أو قلة النوم؟'),
            )),
            Question('q9', 'Has your daily energy decreased even if you sleep well?', 'هل طاقتك اليومية قلت حتى لو بتنام كويس؟', (
                Item('q9a', 'Do you feel mental exhaustion (not just physical fatigue)?', 'هل بتحس بإرهاق ذهني (مش بس تعب جسدي)؟'),
                Item('q9b', 'Does fatigue increase in hot weather?', 'هل التعب بيزيد مع الجو الحار؟'),
            )),
        ),
    ),
    Section(
        key='pain',
        title_en='Section 5: Pain and Body Control',
        title_ar='القسم 5: الألم والتحكم في الجسم',
        max_score=10,
        questions=(
            Question('q10', 'Do you feel strange pain (like electric shock or burning) without clear reason?', 'هل بتحس بألم غريب (زي لسعة كهربا أو حرقان) بدون سبب واضح؟', (
                Item('q10a', 'Does the pain appear in the same area?', 'هل الألم بيظهر في نفس المنطقة؟'),
                Item('q10b', 'Does it increase with movement or touching the skin?', 'هل بيزيد مع الحركة أو لمس الجلد؟'),
            )),
            Question('q11', 'Do you have difficulty controlling urination or feel a sudden need to urinate?', 'هل بتواجه صعوبة في التحكم في البول أو بتحس بحاجة مفاجئة للتبول؟', (
                Item('q11a', 'Does it happen frequently?', 'هل بتحصل بشكل متكرر؟'),
                Item('q11b', 'Is it accompanied by numbness in the lower back or abdomen?', 'هل بتصاحبها تنميل أسفل الظهر أو البطن؟'),
            )),
        ),
    ),
    Section(
        key='mood',
        title_en='Section 6: Mood and Sleep',
        title_ar='القسم 6: المزاج والنوم',
        max_score=10,
        questions=(
            Question('q12', 'Does your mood change quickly or do you feel depressed or anxious lately?', 'هل مزاجك بيتغير بسرعة أو بتحس باكتئاب أو قلق مؤخرًا؟', (
                Item('q12a', 'Did these changes start with physical symptoms?', 'هل التغيرات دي بدأت مع الأعراض الجسدية؟'),
                Item('q12b', 'Do you feel your sleep has been affected?', 'هل بتحس إن نومك اتأثر؟'),
            )),
            Question('q13', 'Has your sleep become interrupted or do you wake up several times at night without reason?', 'هل نومك بقى متقطع أو بتصحى أكتر من مرة في الليل بدون سبب؟', (
                Item('q13a', 'Is the reason pain, numbness, or leg cramps?', 'هل السبب ألم أو تنميل أو تقلصات في الرجل؟'),
                Item('q13b', 'Do you feel tired the next day even if you slept enough?', 'هل بتحس بتعب في اليوم التالي حتى لو نمت كفاية؟'),
            )),
        ),
    ),
    Section(
        key='history',
        title_en='Section 7: Medical and Family History',
        title_ar='القسم 7: التاريخ المرضي والعائلة',
        max_score=10,
        questions=(
            Question('q14', 'Is anyone in your family diagnosed with Multiple Sclerosis or autoimmune diseases?', 'هل في حد من عيلتك مصاب بمرض التصلّب المتعدد أو أمراض مناعة؟', (
                Item('q14a', 'First-degree relative? (father, mother, brother, sister)', 'قريب من الدرجة الأولى؟ (أب، أم، أخ، أخت)'),
                Item('q14b', 'Second-degree relative? (grandfather, grandmother, uncle, aunt)', 'قريب من الدرجة الثانية؟ (جد، جدة، عم، عمة، خال، خالة)'),
            )),
            Question('q15', 'Have you previously suffered from recurrent infections or other autoimmune diseases?', 'هل سبق وعانيت من التهابات متكررة أو أمراض مناعة أخرى؟', (
                Item('q15a', 'Were the infections in the eye or nerves?', 'هل كانت الالتهابات في العين أو الأعصاب؟'),
                Item('q15b', 'Did the symptoms persist for a long time or recur more than once?', 'هل استمرت الأعراض لفترة طويلة أو تكررت أكثر من مرة؟'),
            )),
        ),
    ),
)

_ALL_IDS = frozenset(
    id_
    for section in SECTIONS
    for question in section.questions
    for id_ in (question.id, *(item.id for item in question.follow_ups))
)


def level_for(points: int, max_points: int) -> str:
    share = points / max_points
    if share >= HIGH_SHARE:
        return "high"
    if share >= MEDIUM_SHARE:
        return "medium"
    return "low"


def get_questions() -> dict:
    """The survey layout with bilingual texts, for the frontend to render."""
    return {
        "sections": [
            {
                "key": section.key,
                "title_en": section.title_en,
                "title_ar": section.title_ar,
                "questions": [
                    {
                        "id": question.id,
                        "text_en": question.text_en,
                        "text_ar": question.text_ar,
                        "follow_ups": [
                            {"id": item.id, "text_en": item.text_en, "text_ar": item.text_ar}
                            for item in question.follow_ups
                        ],
                    }
                    for question in section.questions
                ],
            }
            for section in SECTIONS
        ],
        "note_en": NOTE_EN,
        "note_ar": NOTE_AR,
    }


def evaluate(answers: dict[str, bool]) -> dict:
    """Score the answers. Ids not sent count as "no"; follow-ups only count under a "yes"."""
    unknown = sorted(set(answers) - _ALL_IDS)
    if unknown:
        raise SurveyInputError(f"unknown question id(s): {', '.join(unknown)}")

    sections, total, max_total = [], 0, 0
    for section in SECTIONS:
        points = 0
        for question in section.questions:
            if answers.get(question.id):
                points += MAIN_POINTS
                points += ITEM_POINTS * sum(bool(answers.get(item.id)) for item in question.follow_ups)
        sections.append({
            "key": section.key,
            "title_en": section.title_en,
            "title_ar": section.title_ar,
            "points": points,
            "max_points": section.max_score,
            "level": level_for(points, section.max_score),
        })
        total += points
        max_total += section.max_score

    level = level_for(total, max_total)
    label_en, label_ar = LEVEL_LABELS[level]
    return {
        "symptom_score": round(100.0 * total / max_total, 1),
        "level": level,
        "label_en": label_en,
        "label_ar": label_ar,
        "points": total,
        "max_points": max_total,
        "sections": sections,
        "note_en": NOTE_EN,
        "note_ar": NOTE_AR,
    }
