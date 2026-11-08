package de.duecki.calibraw;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;

import java.nio.ByteBuffer;

/** Draws an emoji with Android's colour emoji font, for the triage card's badge. */
final class EmojiRenderer {
    private EmojiRenderer() {}

    /**
     * {@code emoji} centred in a {@code sizePx} square, as premultiplied RGBA rows, or an empty
     * array when there is nothing to draw or the font has no glyph for it.
     */
    static byte[] draw(String emoji, int sizePx) {
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        if (emoji == null || emoji.isEmpty() || sizePx <= 0 || !paint.hasGlyph(emoji)) {
            return new byte[0];
        }
        paint.setTextSize(sizePx * 0.8f);
        paint.setTextAlign(Paint.Align.CENTER);
        Paint.FontMetrics metrics = paint.getFontMetrics();
        Bitmap bitmap = Bitmap.createBitmap(sizePx, sizePx, Bitmap.Config.ARGB_8888);
        new Canvas(bitmap).drawText(
                emoji, sizePx / 2f, sizePx / 2f - (metrics.ascent + metrics.descent) / 2f, paint);
        // ARGB_8888 stores premultiplied bytes in R, G, B, A order.
        ByteBuffer pixels = ByteBuffer.allocate(sizePx * sizePx * 4);
        bitmap.copyPixelsToBuffer(pixels);
        bitmap.recycle();
        return pixels.array();
    }
}
