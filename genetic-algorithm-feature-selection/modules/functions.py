import logging

logging.basicConfig(level=logging.DEBUG)

def chain_safe(*iterables):
  for iterable in iterables:
    yield from iterable

def chain(*iterables):
    for iterable in iterables:
        try:
            yield from iterable
        except TypeError as e:
            print(e)
