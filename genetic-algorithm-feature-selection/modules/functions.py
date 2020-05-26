def chain(*iterables):
  for iterable in iterables:
    yield from iterable
