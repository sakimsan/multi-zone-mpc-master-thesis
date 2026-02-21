
import re


def check_characters_in_front_of_numbers(filepath):
    with open(filepath, "r") as file:
        content = file.read()

    def find_si_cases(input_string, start=0, matches=None):
        if matches is None:
            matches = []

        search_string = r'\SI'
        pos = input_string.find(search_string, start)

        # Base case: if no more occurrences of '\SI' are found
        if pos == -1:
            return matches

        # Check if there is a character before '\SI' and it is not '~' or a space
        if pos > 0 and input_string[pos - 1] not in ('~', ' ', '<', '>', '-', '(', '/'):
            matches.append(input_string[pos - 5:pos + len(search_string)])

        # Recursive call to find the next occurrence
        return find_si_cases(input_string, pos + 1, matches)

    # Example usage
    result = find_si_cases(content)
    print(result)


if __name__ == '__main__':
    check_characters_in_front_of_numbers(r"D:\full_paper.tex")
