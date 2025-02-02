from tasks import register_task, TaskType
from tasks.base_selection import BaseProbTaskManager
import re

extract_re = re.compile(r"^\((.+?)\)\w*(.+?)$")


def process_example(ex):
    # ex: {"input", "target"}
    # where `input` may end with SomeRandomContext\nOptions:\n(A) TextForChoiceA\n(B) TextForChoiceB\n(C) ...'
    # and `target` may looks like (A) or (B)
    # we then extract and return SomeRandomContext, [TextForChoiceA, TextForChoiceB, ...] and `target_answer` , which stands for the golden choice
    query_with_choices, target = ex["input"], ex["target"]
    query, rest = query_with_choices.rsplit("\nOptions:\n")  # make sure cut at right-most position
    rest = rest.split("\n")
    choices = []
    target_answer = None
    for line in rest:
        matched = extract_re.match(line)
        if not matched:
            continue
        groups = matched.groups()
        if len(groups) != 2:
            continue
        answer_key, answer = [e.strip() for e in groups]
        choices.append(answer)
        if answer_key in target:
            target_answer = answer
    return query, choices, target_answer


class BbhBaseProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = None
        self.can_be_stratified = None
        self.num_base_shot = None

    def promptify_input(self, query):
        with_query = f"Q: {query}\nA:"
        return with_query

    def promptify_input_with_choice(self, query, choice):
        with_query_and_choice = f"{self.promptify_input(query)} {choice}"
        return with_query_and_choice

    def dataset_preprocess(self, raw_data):
        # we perform several forced check to make sure this super-class function
        # is only called from multiple-choice sub-tasks of BBH, not classification tasks
        assert self.CHOICES is None
        assert not self.can_be_stratified
        assert self.num_base_shot == 1
        data = []
        for e in raw_data:
            query, choices, target_answer = process_example(e)
            if len(choices) != self.expected_num_choices:
                continue
            data.append({"query": query.strip(), "choices": choices, "answer_idx": choices.index(target_answer)})
        return data


@register_task(name="boolean_expressions", task_type=TaskType.Selection, suite=["bbh", "classification"])
class BbhBooleanExpressionsProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["False", "True"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "boolean_expressions", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            ("not ( ( not not True ) )", "False"),
            ("True and False and not True and True", "False"),
            ("not not ( not ( False ) )", "True"),
        ]
        return selected_examples

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            answer_idx = self.CHOICES.index(e["target"])
            data.append({"query": e["input"].strip(), "choices": self.CHOICES, "answer_idx": answer_idx})
        return data


@register_task(name="causal_judgement", task_type=TaskType.Selection, suite=["bbh", "classification"])
class BbhCausalJudegmentProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["Yes", "No"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "causal_judgement", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                "How would a typical person answer each of the following questions about causation?\nFrank T., had an ongoing dispute with his neighbor over a stretch of land and one day decided to shoot his neighbor in the body. Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild. Nonetheless, the bullet bounced off a large boulder several feet away and hit the neighbor's body, causing significant injury. Did Frank T. intentionally shoot his neighbor in the body?",
                "No",
            ),
            (
                'How would a typical person answer each of the following questions about causation?\nSuzy and Billy are working on a project that is very important for our nation\'s security. The boss tells them both: "Be sure that you are here at exactly 9 am. It is absolutely essential that you arrive at that time." Both Billy and Suzy arrive at 9 am. As it happens, there was a motion detector installed in the room where they arrived. The motion detector was set up to be triggered if at least one person appeared in the room at the same time. So the motion detector went off. Did Billy cause the motion detector to go off?',
                "Yes",
            ),
            (
                "How would a typical person answer each of the following questions about causation?\nGeorge and his sister Lena reunite at their parents' house for Thanksgiving. Whereas George just got into medical school, Lena is unhappy in her marriage and recently lost her job. Over the course of the day, George and Lena get into a number of heated arguments. Later in the afternoon they play a game of darts. They split the first two games, and the third game is close until the end. Who will win comes down to George's last shot. If he hits a high point region, he wins; if he hits a low point region, Lena wins. George thinks of the difficult time Lena is having, and he really wants to let her win. He aims the dart at the low point region. He sets up his shot and the dart lands in the low point region. After his shot, Lena wins the game and is very happy. Did George hit the low point region intentionally?",
                "Yes",
            ),
        ]
        return selected_examples

    def dataset_preprocess(self, raw_data):
        def strip_end(text, suffix):
            if suffix and text.endswith(suffix):
                return text[: -len(suffix)]
            return text

        data = []
        for e in raw_data:
            query = strip_end(e["input"].strip(), "Options:\n- Yes\n- No")
            answer_idx = self.CHOICES.index(e["target"])
            data.append({"query": query.strip(), "choices": self.CHOICES, "answer_idx": answer_idx})
        return data


@register_task(name="date_understanding", task_type=TaskType.Selection, suite=["bbh", "multiple-choice"])
class BbhDateUnderstandingProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 6

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "date_understanding", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            ("Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?", "12/14/1937"),
            ("Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?", "11/11/2018"),
            ("Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. What is the date tomorrow in MM/DD/YYYY?", "01/03/1963"),
        ]
        return selected_examples


@register_task(name="disambiguation_qa", task_type=TaskType.Selection, suite=["bbh", "multiple-choice"])
class BbhDisambiguationQAProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 3

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "disambiguation_qa", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The chief told the counselor that they took the day off.",
                "The chief took the day off",
            ),
            (
                "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The manager sent a message to the secretary, but he didn't reply yet.",
                "The secretary didn't reply yet",
            ),
            (
                "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: Bailey will plan to meet the director at his office",
                "Ambiguous",
            ),
        ]
        return selected_examples


@register_task(name="formal_fallacies", task_type=TaskType.Selection, suite=["bbh", "classification"])
class BbhCausaFormalFallaciesProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["invalid", "valid"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "formal_fallacies", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                '"It is not always easy to see who is related to whom -- and in which ways. The following argument pertains to this question: To begin with, Lesley is a close friend of Fernando. Moreover, being a close friend of Fernando or a schoolmate of Lowell is sufficient for being a great-grandfather of Leroy. It follows that Lesley is a great-grandfather of Leroy."\nIs the argument, given the explicitly stated premises, deductively valid or invalid?',
                "valid",
            ),
            (
                '"It is not always easy to see who is related to whom -- and in which ways. The following argument pertains to this question: Whoever is not a great-grandfather of Clyde is a stepbrother of Brian. Being an ancestor of Dana is sufficient for not being a great-grandfather of Clyde. We may conclude: Everyone who is an ancestor of Dana is a stepbrother of Brian, too."\nIs the argument, given the explicitly stated premises, deductively valid or invalid?',
                "valid",
            ),
            (
                '"It is not always easy to grasp who is consuming which products. The following argument pertains to this question: Every infrequent user of Paul Mitchell shampoo is either a rare consumer of Nioxin shampoo or a loyal buyer of Caress soap, or both. No regular consumer of Lush soap is a rare consumer of Nioxin shampoo and, in the same time, a loyal buyer of Caress soap. It follows that whoever is an infrequent user of Paul Mitchell shampoo is not a regular consumer of Lush soap."\nIs the argument, given the explicitly stated premises, deductively valid or invalid?',
                "invalid",
            ),
        ]
        return selected_examples

    def dataset_preprocess(self, raw_data):
        def strip_end(text, suffix):
            if suffix and text.endswith(suffix):
                return text[: -len(suffix)]
            return text

        data = []
        for e in raw_data:
            query = strip_end(e["input"].strip(), "Options:\n- valid \n- invalid")
            answer_idx = self.CHOICES.index(e["target"])
            data.append({"query": query.strip(), "choices": self.CHOICES, "answer_idx": answer_idx})
        return data


@register_task(name="geometric_shapes", task_type=TaskType.Selection, suite=["bbh", "multiple-choice"])
class BbhGenmetricShapesProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 10

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "geometric_shapes", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                'This SVG path element <path d="M 31.00,73.00 L 32.00,59.00 L 44.00,50.00 L 49.00,41.00 L 64.00,37.00 L 71.00,55.00 L 64.00,76.00 L 52.00,61.00 L 31.00,73.00"/> draws a',
                "octagon",
            ),
            (
                'This SVG path element <path d="M 14.19,26.04 L 51.43,39.21 L 58.44,36.69 L 56.63,30.17 L 48.53,26.66 L 14.19,26.04"/> draws a',
                "pentagon",
            ),
            (
                'This SVG path element <path d="M 41.00,43.00 L 37.00,34.00 L 41.00,33.00 L 45.00,34.00 L 41.00,43.00"/> draws a',
                "kite",
            ),
        ]
        return selected_examples


@register_task(name="hyperbaton", task_type=TaskType.Selection, suite=["bbh", "multiple-choice"])
class BbhHyperbatonProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 2

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "hyperbaton", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                "Which sentence has the correct adjective order:",
                "terrible rubber ship",
            ),
            (
                "Which sentence has the correct adjective order:",
                "repulsive small Brazilian exercise ship",
            ),
            (
                "Which sentence has the correct adjective order:",
                "wonderful square blue gold shoe",
            ),
        ]
        return selected_examples


@register_task(name="logical_deduction_three_objects", task_type=TaskType.Selection, suite=["bbh", "multiple-choice", "logical_deduction"])
class BbhLogicalDeductionThreeObjectsProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 3

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "logical_deduction_three_objects", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                "The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. In a golf tournament, there were three golfers: Amy, Eli, and Eve. Eve finished above Amy. Eli finished below Amy.",
                "Eli finished last",
            ),
            (
                "The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. On a shelf, there are three books: a white book, a green book, and an orange book. The green book is to the right of the white book. The orange book is the rightmost.",
                "The white book is the leftmost",
            ),
            (
                "The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. On a shelf, there are three books: a red book, a gray book, and a white book. The white book is to the left of the gray book. The red book is the second from the left.",
                "The white book is the leftmost",
            ),
        ]
        return selected_examples


@register_task(name="logical_deduction_five_objects", task_type=TaskType.Selection, suite=["bbh", "multiple-choice", "logical_deduction"])
class BbhLogicalDeductionFiveObjectsProbTask(BbhLogicalDeductionThreeObjectsProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 5
        # logical_deduction_five_objects using the same examples as logical_deduction_three_objects

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "logical_deduction_five_objects", "train"),
            "sampling": None,
        }


@register_task(name="logical_deduction_seven_objects", task_type=TaskType.Selection, suite=["bbh", "multiple-choice", "logical_deduction"])
class BbhLogicalDeductionSevenObjectsProbTask(BbhLogicalDeductionThreeObjectsProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 7
        # logical_deduction_seven_objects using the same examples as logical_deduction_three_objects

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "logical_deduction_seven_objects", "train"),
            "sampling": None,
        }


@register_task(name="movie_recommendation", task_type=TaskType.Selection, suite=["bbh", "multiple-choice"])
class BbhMovieRecommendationProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 4

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "movie_recommendation", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                "Find a movie similar to Star Wars Episode IV - A New Hope, Indiana Jones and the Last Crusade, Star Wars Episode V - The Empire Strikes Back, The Big Lebowski:",
                "The Princess Bride",
            ),
            ("Find a movie similar to Twister, The Silence of the Lambs, Independence Day, Braveheart:", "Forrest Gump"),
            ("Find a movie similar to Minority Report, Total Recall, Inside Out, Forrest Gump:", "Edge of Tomorrow"),
        ]
        return selected_examples


@register_task(name="navigate", task_type=TaskType.Selection, suite=["bbh", "classification"])
class BbhNavigateProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["No", "Yes"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "navigate", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                "If you follow these instructions, do you return to the starting point? Turn left. Turn around. Turn left. Take 7 steps. Take 2 steps. Take 4 steps. Take 8 steps.",
                "No",
            ),
            (
                "If you follow these instructions, do you return to the starting point? Turn around. Take 1 step. Take 6 steps. Turn around. Take 6 steps. Take 9 steps. Take 1 step.",
                "No",
            ),
            (
                "If you follow these instructions, do you return to the starting point? Always face forward. Take 2 steps right. Take 9 steps left. Take 7 steps right.",
                "Yes",
            ),
        ]
        return selected_examples

    def dataset_preprocess(self, raw_data):
        def strip_end(text, suffix):
            if suffix and text.endswith(suffix):
                return text[: -len(suffix)]
            return text

        data = []
        for e in raw_data:
            query = strip_end(e["input"].strip(), "Options:\n- Yes\n- No")
            answer_idx = self.CHOICES.index(e["target"])
            data.append({"query": query.strip(), "choices": self.CHOICES, "answer_idx": answer_idx})
        return data


@register_task(name="penguins_in_a_table", task_type=TaskType.Selection, suite=["bbh", "multiple-choice"])
class BbhPenguinsInATableProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 5

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "penguins_in_a_table", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                "Here is a table where the first line is a header and each subsequent line is a penguin:  name, age, height (cm), weight (kg) Louis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15  For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.  We now add a penguin to the table:\nJames, 12, 90, 12\nHow many penguins are less than 8 years old?",
                "2",
            ),
            (
                "Here is a table where the first line is a header and each subsequent line is a penguin:  name, age, height (cm), weight (kg) Louis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15  For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.  Which is the youngest penguin?",
                "Bernard",
            ),
            (
                "Here is a table where the first line is a header and each subsequent line is a penguin:  name, age, height (cm), weight (kg) Louis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15  For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.  What is the name of the second penguin sorted by alphabetic order?",
                "Gwen",
            ),
        ]
        return selected_examples


@register_task(name="ruin_names", task_type=TaskType.Selection, suite=["bbh", "multiple-choice"])
class BbhRuinNamesProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 4

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "ruin_names", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                "Which of the following is a humorous edit of this artist or movie name: 'whitesnake'?",
                "whitesnape",
            ),
            (
                "Which of the following is a humorous edit of this artist or movie name: 'one of our dinosaurs is missing'?",
                "one of our dinosaurs is pissing",
            ),
            (
                "Which of the following is a humorous edit of this artist or movie name: 'counting crows'?",
                "counting cows",
            ),
        ]
        return selected_examples


@register_task(name="salient_translation_error_detection", task_type=TaskType.Selection, suite=["bbh", "multiple-choice"])
class BbhSalientTranslationErrorDetectionProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 6

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "salient_translation_error_detection", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                "The following translations from German to English contain a particular error. That error will be one of the following types: Named Entities: An entity (names, places, locations, etc.) is changed to a different entity. Numerical Values: Numerical values (ordinals or cardinals), dates, and/or units are changed. Modifiers or Adjectives: The modifiers and adjectives pertaining to a noun are changed. Negation or Antonyms: Introduce or remove a negation or change comparatives to their antonyms. Facts: Trivial factual errors not pertaining to the above classes are introduced in the translations. Dropped Content: A significant clause in the translation is removed. Please identify that error.  Source: In der Liste der Baudenkmale in Lenzen (Elbe) sind alle Baudenkmale der brandenburgischen Stadt Lenzen (Elbe) und ihrer Ortsteile aufgelistet.\nTranslation: In the list of architectural monuments in Lenzen all architectural monuments of the Brandenburg city of Lenzen and its districts are listed.\nThe translation contains an error pertaining to",
                "Named Entities",
            ),
            (
                "The following translations from German to English contain a particular error. That error will be one of the following types: Named Entities: An entity (names, places, locations, etc.) is changed to a different entity. Numerical Values: Numerical values (ordinals or cardinals), dates, and/or units are changed. Modifiers or Adjectives: The modifiers and adjectives pertaining to a noun are changed. Negation or Antonyms: Introduce or remove a negation or change comparatives to their antonyms. Facts: Trivial factual errors not pertaining to the above classes are introduced in the translations. Dropped Content: A significant clause in the translation is removed. Please identify that error.  Source: Auf dieser Seite sind die Baudenkmäler der oberbayerischen Großen Kreisstadt Landsberg am Lech zusammengestellt.\nTranslation: On this page are compiled the architectural monuments of the town of Landsberg am Lech.\nThe translation contains an error pertaining to",
                "Dropped Content",
            ),
            (
                "The following translations from German to English contain a particular error. That error will be one of the following types: Named Entities: An entity (names, places, locations, etc.) is changed to a different entity. Numerical Values: Numerical values (ordinals or cardinals), dates, and/or units are changed. Modifiers or Adjectives: The modifiers and adjectives pertaining to a noun are changed. Negation or Antonyms: Introduce or remove a negation or change comparatives to their antonyms. Facts: Trivial factual errors not pertaining to the above classes are introduced in the translations. Dropped Content: A significant clause in the translation is removed. Please identify that error.  Source: Łeba ist eine Kleinstadt und ein Badeort im Powiat Lęborski der polnischen Woiwodschaft Pommern.\nTranslation: Eba is not a small town and seaside resort in the Powiat Léborski county of the Pomeranian Voivodeship of Poland.\nThe translation contains an error pertaining to",
                "Negation or Antonyms",
            ),
        ]
        return selected_examples


@register_task(name="snarks", task_type=TaskType.Selection, suite=["bbh", "multiple-choice"])
class BbhSnarksProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 2

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "snarks", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            ("Which statement is sarcastic?", "Yes, because having interests and actively researching them is a huge waste"),
            ("Which statement is sarcastic?", "No one is going to disagree with you on this. Ad hominem attacks really help your case"),
            ("Which statement is sarcastic?", "Consistency in the league's punishments? What do you think this is supposed to be, politics?"),
        ]
        return selected_examples


@register_task(name="sports_understanding", task_type=TaskType.Selection, suite=["bbh", "classification"])
class BbhSportsUnderstandingProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["no", "yes"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "sports_understanding", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            ('Is the following sentence plausible? "Bam Adebayo scored a reverse layup in the Western Conference Finals."', "yes"),
            ('Is the following sentence plausible? "Santi Cazorla scored a touchdown."', "no"),
            ('Is the following sentence plausible? "DeMar DeRozan was called for the goal tend."', "yes"),
        ]
        return selected_examples

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            answer_idx = self.CHOICES.index(e["target"])
            data.append({"query": e["input"].strip(), "choices": self.CHOICES, "answer_idx": answer_idx})
        return data


@register_task(name="temporal_sequences", task_type=TaskType.Selection, suite=["bbh", "multiple-choice"])
class BbhTemporalSequencesProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 4

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "temporal_sequences", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                "Today, Emily went to the museum. Between what times could they have gone?\nWe know that:\nEmily woke up at 1pm.\nElizabeth saw Emily reading at the library from 2pm to 4pm.\nJessica saw Emily watching a movie at the theater from 4pm to 5pm.\nLeslie saw Emily waiting at the airport from 5pm to 6pm.\nWilliam saw Emily buying clothes at the mall from 6pm to 7pm.\nThe museum was closed after 7pm.\nBetween what times could Emily have gone to the museum?",
                "1pm to 2pm",
            ),
            (
                "Today, Elizabeth went to the amusement park. Between what times could they have gone?\nWe know that:\nElizabeth woke up at 7am.\nDavid saw Elizabeth fixing their computer at the electronic store from 1pm to 2pm.\nSarah saw Elizabeth playing tennis at the tennis court from 2pm to 3pm.\nSusan saw Elizabeth walking towards the Statue of Liberty from 3pm to 6pm.\nAndrew saw Elizabeth taking photos near the Eiffel Tower from 6pm to 9pm.\nEmily saw Elizabeth getting a coffee at the cafe from 9pm to 10pm.\nThe amusement park was closed after 10pm.\nBetween what times could Elizabeth have gone to the amusement park?",
                "7am to 1pm",
            ),
            (
                "Today, Tiffany went to the beach. Between what times could they have gone?\nWe know that:\nTiffany woke up at 5am.\nBetty saw Tiffany getting a coffee at the cafe from 5am to 6am.\nJessica saw Tiffany working at the office from 6am to 9am.\nJohn saw Tiffany stretching at a yoga studio from 9am to 12pm.\nSean saw Tiffany sitting on a rooftop from 12pm to 2pm.\nSarah saw Tiffany playing tennis at the tennis court from 2pm to 3pm.\nThe beach was closed after 4pm.\nBetween what times could Tiffany have gone to the beach?",
                "3pm to 4pm",
            ),
        ]
        return selected_examples


@register_task(
    name="tracking_shuffled_objects_three_objects",
    task_type=TaskType.Selection,
    suite=["bbh", "multiple-choice", "tracking_shuffled_objects"],
)
class BbhTrackingShuffledObjectsThreeObjectsProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 3

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "tracking_shuffled_objects_three_objects", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                "Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a yellow ball, Bob has a blue ball, and Claire has a pink ball.\nAs the game progresses, pairs of players trade balls. First, Claire and Alice swap balls. Then, Alice and Bob swap balls. Finally, Claire and Bob swap balls. At the end of the game, Bob has the",
                "yellow ball",
            ),
            (
                "Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a white ball, Bob has a purple ball, and Claire has a pink ball.\nAs the game progresses, pairs of players trade balls. First, Bob and Alice swap balls. Then, Bob and Claire swap balls. Finally, Bob and Alice swap balls. At the end of the game, Alice has the",
                "pink ball",
            ),
            (
                "Alice, Bob, and Claire are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Lola, Bob is dancing with Rodrigo, and Claire is dancing with Patrick.\nThroughout the song, the dancers often trade partners. First, Alice and Bob switch partners. Then, Claire and Bob switch partners. Finally, Bob and Alice switch partners. At the end of the dance, Alice is dancing with",
                "Patrick",
            ),
        ]
        return selected_examples


@register_task(
    name="tracking_shuffled_objects_five_objects",
    task_type=TaskType.Selection,
    suite=["bbh", "multiple-choice", "tracking_shuffled_objects"],
)
class BbhTrackingShuffledObjectsFiveObjectsProbTask(BbhTrackingShuffledObjectsThreeObjectsProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 5

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "tracking_shuffled_objects_five_objects", "train"),
            "sampling": None,
        }


@register_task(
    name="tracking_shuffled_objects_seven_objects",
    task_type=TaskType.Selection,
    suite=["bbh", "multiple-choice", "tracking_shuffled_objects"],
)
class BbhTrackingShuffledObjectsSevenObjectsProbTask(BbhTrackingShuffledObjectsThreeObjectsProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1
        self.expected_num_choices = 7

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "tracking_shuffled_objects_seven_objects", "train"),
            "sampling": None,
        }


@register_task(name="web_of_lies", task_type=TaskType.Selection, suite=["bbh", "classification"])
class BbhWebOfLiesProbTask(BbhBaseProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["No", "Yes"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("maveriq/bigbenchhard", "web_of_lies", "train"),
            "sampling": None,
        }

    def task_prepared_examplars(self):
        selected_examples = [
            (
                "Question: Fidel tells the truth. Jerry says Fidel tells the truth. Vina says Jerry tells the truth. Millicent says Vina lies. Raymond says Millicent lies. Does Raymond tell the truth?",
                "Yes",
            ),
            (
                "Question: Kristian lies. Millie says Kristian lies. Maybelle says Millie tells the truth. Fidel says Maybelle lies. Leda says Fidel lies. Does Leda tell the truth?",
                "Yes",
            ),
            (
                "Question: Kristian tells the truth. Michaela says Kristian lies. Raymond says Michaela tells the truth. Osvaldo says Raymond tells the truth. Jamey says Osvaldo tells the truth. Does Jamey tell the truth?",
                "No",
            ),
        ]
        return selected_examples

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            answer_idx = self.CHOICES.index(e["target"])
            data.append({"query": e["input"].strip(), "choices": self.CHOICES, "answer_idx": answer_idx})
        return data
