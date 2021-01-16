package prover;

import static org.junit.Assert.*;

import org.junit.Test;

public class TptpParserTest {
  @Test
  public void read() {
    var matcher =
        TptpParser.RATING_PATTERN.matcher(
            " Rating   : 0.09 v7.4.0, 0.10 v7.2.0, 0.07 v7.1.0, 0.13 v7.0.0, 0.10 v6.4.0,");
    assertTrue(matcher.lookingAt());
  }
}
