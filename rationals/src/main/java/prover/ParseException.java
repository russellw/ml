package prover;

import java.io.IOException;

public final class ParseException extends IOException {
    private static final long serialVersionUID = 0;
    public final String file;
    public final int line;
    public final String message;

    public ParseException(String file, int line, String message) {
        this.file = file;
        this.line = line;
        this.message = message;
    }

    @Override
    public String getMessage() {
        return String.format("%s:%d: %s", file, line, message);
    }

    @Override
    public String toString() {
        return getClass().getName() + ": " + getMessage();
    }
}
