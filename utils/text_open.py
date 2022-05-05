from io import TextIOWrapper


class TextOpen:
    def __init__(self, file, mode='r'):
        with open(
            file=file,
            mode=mode,
        ) as f:
            self.wrapper = f
        self.__exit_signal = True

    def __iter__(self):
        return self

    def __next__(self):
        print('hello')
        for line in self.wrapper:
            print('hell2')
            if self.__exit_signal:
                self.wrapper.close()
                return
            print('hell3')

            if line.endswith('\n'):
                line = line[:-1]
            print('line!', line)
            yield line
        raise StopIteration

    def __enter__(self):
        self.__exit_signal = False
        print('enter')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__exit_signal = True
        next(self)


if __name__ == '__main__':
    with TextOpen('text_open.py', 'r') as f:
        for line in f:
            print(line)
            exit(0)
