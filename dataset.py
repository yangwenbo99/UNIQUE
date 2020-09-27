import scipy.stats

class AutomatedDataset:
    def __init__(
            self,
            name: str,
            loader):
        self._name = name
        self._loader = loader

    def eval(self, device, model):
        """
        @param model: a function receiving a variable and return a single
                      score value

        $return: srcc, plcc
        """
        q_mos = []
        q_hat = []
        for step, sample_batched in enumerate(self._loader, 0):
            x, y = sample_batched['I'], sample_batched['mos']
            #!x = Variable(x)  # No need in modern release
            x = x.to(device)

            y_bar = model(x)
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        plcc = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
        return srcc, plcc

    def __iter__(self):
        for x in self._loader:
            yield x

    def __len__(self):
        return len(self._loader)


