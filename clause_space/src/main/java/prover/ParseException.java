package prover;

import java.io.IOException;

public final class ParseException extends IOException {
  public ParseException(String file, int line, String message) {
    super(String.format("%s:%d: %s", file, line, message));
  }
}
