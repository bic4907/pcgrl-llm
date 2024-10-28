
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
        self._message = None  # 실제 값을 저장할 내부 변수
        self.message = message  # property를 통해 설정
        self.time = 'Run-time error while executing reward function on the environment'
        # message property 설정

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, value):

        value = str(value)

        # 특정 문자열이 포함되어 있으면 짧은 메시지로 설정
        if "This value became a tracer due to JAX operations on these lines:" in value:
            message = value.split("This value became a tracer due to JAX operations on these lines:")[0]
        else:
            message = value

        if 'Attempted boolean conversion of traced array with shape bool[]' in message:
            message += ('\n\n[Tips]\n- Never use comparison operations (e.g., >, <, ==) along with Jax numpy (jnp.array).'
                        '\n- Never use condition (e.g., if) operation with the jnp.array  (e.g., if jnp.all(sub_array == EMPTY). Use `where` or `cond` instead\n')

        self._message = message

    def __str__(self):
        # split lines
        lined_code = ''

        lines = self.code.split('\n')
        for i, line in enumerate(lines):
            lined_code += f'{i+1} {line}\n'

        return f'\n[Code]\n{lined_code}\n[Message]\n{self.time}\n{self.message}'
