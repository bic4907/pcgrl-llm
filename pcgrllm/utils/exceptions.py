
class RewardParsingException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        self.time = 'Error while parsing reward function from text'

    def __str__(self):
        # split lines
        lined_code = ''

        lines = self.code.split('\n')
        for i, line in enumerate(lines):
            lined_code += f'{i+1} {line}\n'

        return f'\nT[Code]\n{lined_code}\n[Message]\n{self.time}\n{self.message}'


class RewardExecutionException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        self.time = 'Run-time error while executing reward function on the environment'

    def __str__(self):
        # split lines
        lined_code = ''

        lines = self.code.split('\n')
        for i, line in enumerate(lines):
            lined_code += f'{i+1} {line}\n'

        return f'\n[Code]\n{lined_code}\n[Message]\n{self.time}\n{self.message}'
