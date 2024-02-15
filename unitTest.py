import unittest

from Task8Package.Task8 import Task8
class TestTask8(unittest.TestCase):

    def setUp(self):
        self.data = Task8()

    def test_read_or_create_file(self):
        # Test reading from existing file
        content = self.data.read_or_create_file(r"C:\פייתון\נסיון\test_file.txt", "Default content")
        self.assertEqual(content, "Default content")

        # Test creating a new file
        content = self.data.read_or_create_file(r"C:\פייתון\נסיון\new_test_file.txt", "New default content")
        self.assertEqual(content, "New default content")

    def test_read_users_to_generator(self):
        generator = self.data.read_users_to_generator(r"C:\פייתון\נסיון\test_users.txt")
        expected_users = ["Alice", "Bob", "Charlie"]
        for user, expected_user in zip(generator, expected_users):
            self.assertEqual(user, expected_user)

    def test_read_users_to_array(self):
        users = self.data.read_users_to_array(r"C:\פייתון\נסיון\test_users.txt")
        expected_users = ["Alice", "Bob", "Charlie"]
        self.assertEqual(users, expected_users)



    def test_is_exist_name_count_a(self):
        # Test with existing name
        result = self.data.is_exist_name_count_a( "Alice", ["Alice", "Lea", "Rivka"])
        self.assertTrue(result)

        # Test with non-existing name
        result = self.data.is_exist_name_count_a( "Tamar", ["Alice", "Lea", "Rivka"])
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
