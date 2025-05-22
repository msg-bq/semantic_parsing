from torch.utils.data import Dataset


class Example:
    def __init__(self, inp, out, cand_out=None):
        assert out or cand_out

        self.input = inp
        self.output = out
        self._candidate_output: list = cand_out  # fixme: cand_out该不该留

    def __getitem__(self, item):
        if item == 'input':
            return self.input
        elif item == 'output':
            return self.output
        else:
            raise KeyError(f"Invalid key: {item}")

    def __str__(self):
        return f"Input: {self.input}\nOutput: {self.output}"


class CustomDataset(Dataset):
    def __init__(self, data: list[Example]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()
