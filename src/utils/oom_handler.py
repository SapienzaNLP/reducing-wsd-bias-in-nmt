from functools import wraps

import torch


def skip_on_OOM(empty_grad=False):

    def _decorator(method):

        @wraps(method)
        def _impl(self, *method_args, **method_kwargs):

            try:
                return method(self, *method_args, **method_kwargs)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('OOM error, skipping batch')

                    if empty_grad:
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory

                    torch.cuda.empty_cache()
                    return

        return _impl

    return _decorator
