import Vision
import CoreGraphics
import CoreVideo
import Foundation

// Режим «Объект по тапу».
//  1) «Найти на кадре»: Vision выделяет объекты переднего плана, каждый из них
//     показывается пунктирной рамкой. Тап внутри рамки берёт её (это уже умеет приложение).
//  2) Тап по пустому месту: рамка «вырастает» из точки по похожему цвету.
//     Для знаков, столбов и других однотонных вещей это обычно надёжнее.
// Дальше рамку ведёт обычный трекинг.

// MARK: - Vision: объекты переднего плана

enum ForegroundPicker {
    /// Нормализованные прямоугольники (начало снизу слева, как в Vision).
    static func rects(_ image: CGImage) -> [CGRect] {
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        let req = VNGenerateForegroundInstanceMaskRequest()
        do {
            try handler.perform([req])
        } catch {
            return []
        }
        guard let obs = req.results?.first else { return [] }

        let mask = obs.instanceMask
        guard CVPixelBufferGetPixelFormatType(mask) == kCVPixelFormatType_OneComponent8 else { return [] }

        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(mask) else { return [] }

        let w = CVPixelBufferGetWidth(mask)
        let h = CVPixelBufferGetHeight(mask)
        let bpr = CVPixelBufferGetBytesPerRow(mask)
        if w < 2 || h < 2 { return [] }
        let p = base.assumingMemoryBound(to: UInt8.self)

        // Значение пикселя это номер объекта (0 = фон).
        var minX = [Int](repeating: w, count: 256)
        var maxX = [Int](repeating: -1, count: 256)
        var minY = [Int](repeating: h, count: 256)
        var maxY = [Int](repeating: -1, count: 256)
        var cnt = [Int](repeating: 0, count: 256)

        for y in 0..<h {
            let row = p + y * bpr
            for x in 0..<w {
                let l = Int(row[x])
                if l == 0 { continue }
                cnt[l] += 1
                if x < minX[l] { minX[l] = x }
                if x > maxX[l] { maxX[l] = x }
                if y < minY[l] { minY[l] = y }
                if y > maxY[l] { maxY[l] = y }
            }
        }

        let total = w * h
        let minPix = max(4, total / 2000)
        var found: [(count: Int, rect: CGRect)] = []

        for l in 1..<256 where cnt[l] >= minPix {
            let x0 = CGFloat(minX[l]) / CGFloat(w)
            let x1 = CGFloat(maxX[l] + 1) / CGFloat(w)
            let yTop = CGFloat(minY[l]) / CGFloat(h)
            let yBot = CGFloat(maxY[l] + 1) / CGFloat(h)
            let r = CGRect(x: x0, y: 1 - yBot, width: x1 - x0, height: yBot - yTop)
            found.append((cnt[l], pad(r, 0.01)))
        }

        found.sort { $0.count > $1.count }
        return found.prefix(12).map { $0.rect }
    }

    private static func pad(_ r: CGRect, _ k: CGFloat) -> CGRect {
        let x0 = max(0, r.minX - k)
        let y0 = max(0, r.minY - k)
        let x1 = min(1, r.maxX + k)
        let y1 = min(1, r.maxY + k)
        return CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    }
}

// MARK: - Рост области от точки тапа

enum GrowPicker {
    static let smallW = 240

    /// nx, ny: 0...1, начало СВЕРХУ слева (как координаты тапа по экрану).
    /// Результат: прямоугольник с началом снизу слева, либо nil, если ничего не вышло.
    static func rect(image: CGImage, nx: Double, ny: Double) -> CGRect? {
        guard let bm = bitmap(image) else { return nil }
        let px = bm.px
        let w = bm.w
        let h = bm.h

        let cx = min(w - 1, max(0, Int(nx * Double(w))))
        let cy = min(h - 1, max(0, Int(ny * Double(h))))

        // Цвет области под пальцем: среднее по небольшому квадрату.
        var sr = 0.0, sg = 0.0, sb = 0.0, n = 0.0
        for dy in -2...2 {
            for dx in -2...2 {
                let x = min(w - 1, max(0, cx + dx))
                let y = min(h - 1, max(0, cy + dy))
                let o = (y * w + x) * 4
                sb += Double(px[o])
                sg += Double(px[o + 1])
                sr += Double(px[o + 2])
                n += 1
            }
        }
        let seed = (r: sr / n, g: sg / n, b: sb / n)

        let total = w * h
        let maxPix = total / 4          // область больше четверти кадра: скорее всего «утекла»
        let minPix = max(6, total / 3000)

        // Сначала пробуем свободный допуск; если область утекла, ужесточаем.
        for tol in [70.0, 50.0, 36.0, 26.0] {
            if let r = grow(px, w, h, cx, cy, seed, tol, maxPix, minPix) { return r }
        }
        return nil
    }

    private static func grow(_ px: [UInt8], _ w: Int, _ h: Int, _ cx: Int, _ cy: Int,
                             _ seed: (r: Double, g: Double, b: Double),
                             _ tol: Double, _ maxPix: Int, _ minPix: Int) -> CGRect? {
        let tol2 = tol * tol
        var seen = [Bool](repeating: false, count: w * h)
        var stack: [Int] = [cy * w + cx]
        seen[cy * w + cx] = true

        var count = 0
        var minX = w, maxX = 0, minY = h, maxY = 0

        func similar(_ i: Int) -> Bool {
            let b = Double(px[i * 4])
            let g = Double(px[i * 4 + 1])
            let r = Double(px[i * 4 + 2])
            let dr = r - seed.r, dg = g - seed.g, db = b - seed.b
            return dr * dr + dg * dg + db * db <= tol2
        }

        while let i = stack.popLast() {
            if !similar(i) { continue }
            count += 1
            if count > maxPix { return nil }

            let x = i % w
            let y = i / w
            if x < minX { minX = x }
            if x > maxX { maxX = x }
            if y < minY { minY = y }
            if y > maxY { maxY = y }

            if x > 0, !seen[i - 1] { seen[i - 1] = true; stack.append(i - 1) }
            if x < w - 1, !seen[i + 1] { seen[i + 1] = true; stack.append(i + 1) }
            if y > 0, !seen[i - w] { seen[i - w] = true; stack.append(i - w) }
            if y < h - 1, !seen[i + w] { seen[i + w] = true; stack.append(i + w) }
        }

        if count < minPix { return nil }

        let pad: CGFloat = 0.012
        let x0 = max(0, CGFloat(minX) / CGFloat(w) - pad)
        let x1 = min(1, CGFloat(maxX + 1) / CGFloat(w) + pad)
        let yTop = max(0, CGFloat(minY) / CGFloat(h) - pad)
        let yBot = min(1, CGFloat(maxY + 1) / CGFloat(h) + pad)
        return CGRect(x: x0, y: 1 - yBot, width: x1 - x0, height: yBot - yTop)
    }

    /// BGRA, строки сверху вниз.
    private static func bitmap(_ image: CGImage) -> (px: [UInt8], w: Int, h: Int)? {
        let sw = min(smallW, image.width)
        let sh = max(1, sw * image.height / max(1, image.width))
        var small = [UInt8](repeating: 0, count: sw * sh * 4)
        let info = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        let ok = small.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(
                data: buf.baseAddress, width: sw, height: sh, bitsPerComponent: 8,
                bytesPerRow: sw * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info
            ) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: sw, height: sh))
            return true
        }
        return ok ? (small, sw, sh) : nil
    }
}

// MARK: - Редактор

extension Editor {
    /// Тап по объекту: сначала ищем его по маске Vision (точный контур),
    /// если не нашли, рамка вырастает из точки по цвету. Дальше трекинг ведёт её как обычную.
    func pickObject(nx: Double, ny: Double) {
        pause()
        let t = currentTime
        Task { @MainActor in
            guard let cg = await self.frameImage(at: t) else { return }
            let found = await Task.detached { () -> (CGRect, Bool)? in
                let insts = ObjectSeg.instances(image: cg)
                if let o = ObjectSeg.at(insts, nx: nx, ny: ny) { return (o.rect, true) }
                if let r = GrowPicker.rect(image: cg, nx: nx, ny: ny) { return (r, false) }
                return nil
            }.value
            if let f = found {
                self.addBox(f.0)
                if f.1 { self.markObject() }
            } else {
                self.message = "Не получилось выделить объект. Нарисуй рамку пальцем."
            }
        }
    }
}
