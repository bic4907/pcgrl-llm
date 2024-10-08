
class RewardParsingException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message


    def __str__(self):
        # split lines
        lined_code = ''

        lines = self.code.split('\n')
        for i, line in enumerate(lines):
            lined_code += f'{i+1} {line}\n'

        return f'\n[Code]\n{lined_code}\n[Message]\n{self.message}'


class RewardExecutionException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message


    def __str__(self):
        # split lines
        lined_code = ''

        lines = self.code.split('\n')
        for i, line in enumerate(lines):
            lined_code += f'{i+1} {line}\n'

        return f'\n[Code]\n{lined_code}\n[Message]\n{self.message}'
