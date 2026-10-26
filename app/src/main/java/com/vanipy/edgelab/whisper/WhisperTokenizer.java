package com.vanipy.edgelab.whisper;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Decode-only Whisper/GPT-2 byte-level BPE tokenizer, parsed from tokenizer.json.
 */
public class WhisperTokenizer {

    private final Map<Integer, String> idToPiece = new HashMap<>();
    private final Map<String, Integer> specialToId = new HashMap<>();
    private final Set<Integer> specialIds = new HashSet<>();
    private final Map<Character, Integer> reverseByteUnicode;

    public WhisperTokenizer(String tokenizerJsonPath) throws Exception {
        String text = new String(Files.readAllBytes(Paths.get(tokenizerJsonPath)), StandardCharsets.UTF_8);
        JSONObject root = new JSONObject(text);
        JSONObject vocab = root.getJSONObject("model").getJSONObject("vocab");
        java.util.Iterator<String> keys = vocab.keys();
        while (keys.hasNext()) {
            String piece = keys.next();
            idToPiece.put(vocab.getInt(piece), piece);
        }
        JSONArray added = root.getJSONArray("added_tokens");
        for (int i = 0; i < added.length(); i++) {
            JSONObject t = added.getJSONObject(i);
            int id = t.getInt("id");
            String content = t.getString("content");
            idToPiece.put(id, content);
            specialToId.put(content, id);
            if (t.optBoolean("special", false)) specialIds.add(id);
        }
        reverseByteUnicode = buildReverseTable();
    }

    public int specialId(String content, int fallback) {
        Integer id = specialToId.get(content);
        return id != null ? id : fallback;
    }

    public String decode(List<Integer> ids) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (int id : ids) {
            if (specialIds.contains(id)) continue;
            String piece = idToPiece.get(id);
            if (piece == null) continue;
            for (int i = 0; i < piece.length(); i++) {
                Integer b = reverseByteUnicode.get(piece.charAt(i));
                if (b != null) bytes.write(b);
            }
        }
        return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
    }

    /** Inverse of GPT-2's bytes_to_unicode. */
    private static Map<Character, Integer> buildReverseTable() {
        Set<Integer> printable = new HashSet<>();
        for (int i = '!'; i <= '~'; i++) printable.add(i);
        for (int i = 0xA1; i <= 0xAC; i++) printable.add(i);
        for (int i = 0xAE; i <= 0xFF; i++) printable.add(i);
        Map<Character, Integer> rev = new HashMap<>();
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (printable.contains(b)) rev.put((char) b, b);
            else { rev.put((char) (256 + n), b); n++; }
        }
        return rev;
    }
}
