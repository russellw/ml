package prover;

import java.io.IOException;

public final class ParseException extends IOException {
  private static final long serialVersionUID = 0;

  public ParseException(String file, String message) {
    super(String.format("%s: %s", file, message));
  }

  public ParseException(String file, int line, String message) {
    super(String.format("%s:%d: %s", file, line, message));
  }
}
