package prover;

import java.io.PrintWriter;
import java.util.LinkedHashMap;
import java.util.Map;

public abstract class Histogram {
  public final Map<Object, HistogramValue> map = new LinkedHashMap<>();

  public final void inc(Object key) {
    var value = map.get(key);
    if (value == null) {
      map.put(key, new HistogramValueInt(1));
      return;
    }
    map.put(key, new HistogramValueInt(value.val() + 1));
  }

  public abstract String keyHeader();

  public String keyString(Object key) {
    return key.toString();
  }

  public final void print(PrintWriter writer) {
    var keys = map.keySet().toArray();
    sort(keys);
    var max = 0.0;
    for (var key : keys) {
      max = Math.max(max, map.get(key).val());
    }
    writer.println("<tr>");
    writer.printf("<th class=\"padded\" style=\"text-align: left\"><u>%s</u>\n", keyHeader());
    writer.printf("<th class=\"padded\" style=\"text-align: left\"><u>%s</u>\n", valueHeader());
    for (var key : keys) {
      var value = map.get(key);
      writer.println("<tr>");
      writer.println("<td class=\"fixed\">" + keyString(key));
      writer.printf(
          "<td class=\"bar\"><div style=\"background-color: #e0ffe0; width: %f%%\">%s</div>\n",
          value.val() * 100 / max, value);
    }
  }

  public void sort(Object[] keys) {}

  public abstract String valueHeader();
}
