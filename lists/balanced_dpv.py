# code by btilly in answer to
# https://stackoverflow.com/questions/72945293/generating-a-random-string-with-matched-brackets
# modified to generate a list of strings rather than a string of characters
import random


class DPPath:
    def __init__(self):
        self.count = 0
        self.next = None

    def add_option(self, transition, tail):
        if self.next is None:
            self.next = {}
        self.next[transition] = tail
        self.count += tail.count

    def random(self):
        if 0 == self.count:
            return None
        else:
            return self.find(int(random.random() * self.count))

    def find(self, pos):
        result = self._find(pos)
        return list(reversed(result))

    def _find(self, pos):
        if self.next is None:
            return []

        for transition, tail in self.next.items():
            if pos < tail.count:
                result = tail._find(pos)
                result.append(transition)
                return result
            else:
                pos -= tail.count

        raise IndexException("find out of range")


def balanced_dp(n, alphabet):
    # Record that there is 1 empty string with balanced parens.
    base_dp = DPPath()
    base_dp.count = 1

    dps = [base_dp]

    for _ in range(n):
        # We are working backwards towards the start.
        prev_dps = [DPPath()]

        for i in range(len(dps)):
            # prev_dps needs to be bigger in case of closed paren.
            prev_dps.append(DPPath())
            # If there are closed parens, we can open one.
            if 0 < i:
                prev_dps[i - 1].add_option("(", dps[i])

            # alphabet chars don't change paren balance.
            for char in alphabet:
                prev_dps[i].add_option(char, dps[i])

            # Add a closed paren.
            prev_dps[i + 1].add_option(")", dps[i])

        # And we are done with this string position.
        dps = prev_dps

    # Return the one that wound up balanced.
    return dps[0]


# And a quick demo of several random strings.
if __name__ == "__main__":
    for _ in range(10):
        print(balanced_dp(10, ["a", "b", "c"]).random())
