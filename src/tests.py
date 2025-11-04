from unittest import TestCase

from tb_query.core import (
    ValidationError,
    calculate_correlation,
    find_event_files,
    get_all_tags,
    get_tag_statistics,
    get_tag_steps,
    query_tensorboard,
)


class TestTBQuery(TestCase):
    """Pretty much placeholder tests for CI to run and make sure the installed libraries work."""
    def test_find_event_files(self) -> None:
        self.assertEqual(find_event_files("."), {"event_files": []})

    def test_get_all_tags(self) -> None:
        with self.assertRaises(ValidationError):
            get_all_tags("nothing", ["something", "else"])

    def test_get_tag_statistics(self) -> None:
        with self.assertRaises(ValidationError):
            get_tag_statistics("x", ["nothing", "much"])

    def test_get_tag_steps(self) -> None:
        with self.assertRaises(ValidationError):
            get_tag_steps("x", ["nothing", "much"])

    def test_query_tensorboard(self) -> None:
        with self.assertRaises(ValidationError):
            query_tensorboard("x", ["nothing", "much"])

    def test_calculate_correlation(self) -> None:
        self.assertEqual(calculate_correlation({}, {"nothing", "much"}), {})
