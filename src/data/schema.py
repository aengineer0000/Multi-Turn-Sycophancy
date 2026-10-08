from typing import List, Literal
from pydantic import BaseModel, Field


class Turn(BaseModel):
    """Represents a single conversational turn in a dialogue.

    Attributes:
        speaker: The role of the speaker, either 'tutor' or 'student'.
        text: The text utterance spoken by the participant.
    """
    speaker: Literal['tutor', 'student']
    text: str


class CSTutorContrastiveDialogue(BaseModel):
    """Represents a contrastive dialogue item for CS tutoring, structured for mechanistic interpretability.

    The contrastive pair (`final_turn_sycophantic` vs `final_turn_firm`) shares identical preceding
    context (`context_turns`), topic, and student misconception. This structure enables training and
    evaluating layer-wise linear probes (e.g., using Scikit-Learn) to detect representation dynamics
    associated with sycophancy vs factual firmness.

    Attributes:
        topic: The CS topic being discussed (e.g., 'Big-O notation').
        student_misconception: The exact logical error or misconception expressed by the student.
        context_turns: List of preceding Turn objects (e.g., Turn 1: correct tutor prompt/response,
            Turn 2: aggressive pushback by the student with the misconception).
        final_turn_sycophantic: Turn object where the tutor caves to sycophancy and validates
            the student's incorrect logic. Used as part of the contrastive pair for layer-wise linear probes.
        final_turn_firm: Turn object where the tutor holds firm, corrects the student, and maintains
            factual grounding. Used as part of the contrastive pair for layer-wise linear probes.
    """
    topic: str = Field(..., description="The computer science topic under discussion.")
    student_misconception: str = Field(..., description="The exact logical error made by the student.")
    context_turns: List[Turn] = Field(..., description="Preceding context turns leading up to the final contrastive response.")
    final_turn_sycophantic: Turn = Field(
        ...,
        description="Final tutor turn caving to sycophancy. Used as part of a contrastive pair for training layer-wise Scikit-Learn linear probes."
    )
    final_turn_firm: Turn = Field(
        ...,
        description="Final tutor turn staying factually firm. Used as part of a contrastive pair for training layer-wise Scikit-Learn linear probes."
    )
