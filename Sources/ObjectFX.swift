import UIKit
import AVFoundation
import Vision
import CoreGraphics
import CoreVideo
import CoreText
import VideoToolbox
import Foundation

// Точное выделение объекта (человек, коробка, что угодно) и текст, привязанный к нему.
// На каждом кадре Vision отдаёт маску объектов переднего плана. Рамка, которую ведёт
// трекер, говорит, КАКОЙ из них наш. Поверх рисуется контур по маске, а текст
// ложится «на объект» (виден только на объекте) или «за объект» (объект перекрывает буквы).

// MARK: - Один найденный объект

final class ObjInst {
    let w: Int
    let h: Int
    let alpha: [UInt8]      // 0...255, слегка размытая маска, строки сверху вниз
    let rect: CGRect        // нормализованный bbox, начало снизу слева
    let bx0: Int
    let bx1: Int
    let by0: Int
    let by1: Int
    private var cachedMask: CGImage?

    init(w: Int, h: Int, alpha: [UInt8], rect: CGRect, bx0: Int, bx1: Int, by0: Int, by1: Int) {
        self.w = w
        self.h = h
        self.alpha = alpha
        self.rect = rect
        self.bx0 = bx0
        self.bx1 = bx1
        self.by0 = by0
        self.by1 = by1
    }

    /// RGBA-картинка, где прозрачность = маска (для режима destinationIn).
    func maskImage() -> CGImage? {
        if let c = cachedMask { return c }
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0..<(w * h) { rgba[i * 4 + 3] = alpha[i] }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                          bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                          bitmapInfo: info, provider: provider, decode: nil,
                          shouldInterpolate: true, intent: .defaultIntent)
        cachedMask = img
        return img
    }
}

// MARK: - Сегментация (Vision)

enum ObjectSeg {
    static let maxW = 320

    static func instances(pixelBuffer pb: CVPixelBuffer) -> [ObjInst] {
        run(VNImageRequestHandler(cvPixelBuffer: pb, options: [:]))
    }

    static func instances(image: CGImage) -> [ObjInst] {
        run(VNImageRequestHandler(cgImage: image, options: [:]))
    }

    private static func run(_ handler: VNImageRequestHandler) -> [ObjInst] {
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

        let sw = CVPixelBufferGetWidth(mask)
        let sh = CVPixelBufferGetHeight(mask)
        let bpr = CVPixelBufferGetBytesPerRow(mask)
        let step = max(1, sw / maxW)
        let w = sw / step
        let h = sh / step
        if w < 8 || h < 8 { return [] }
        let p = base.assumingMemoryBound(to: UInt8.self)

        var lab = [UInt8](repeating: 0, count: w * h)
        var count = [Int](repeating: 0, count: 256)
        for y in 0..<h {
            let row = p + (y * step) * bpr
            for x in 0..<w {
                let l = row[x * step]
                lab[y * w + x] = l
                if l != 0 { count[Int(l)] += 1 }
            }
        }

        let minPix = max(6, w * h / 1500)
        let labels = (1..<256).filter { count[$0] >= minPix }.sorted { count[$0] > count[$1] }.prefix(8)
        var out: [ObjInst] = []
        for l in labels {
            if let o = makeInst(lab, w, h, UInt8(l)) { out.append(o) }
        }
        return out
    }

    private static func makeInst(_ lab: [UInt8], _ w: Int, _ h: Int, _ l: UInt8) -> ObjInst? {
        var bin = [UInt8](repeating: 0, count: w * h)
        var x0 = w, x1 = -1, y0 = h, y1 = -1
        for y in 0..<h {
            for x in 0..<w where lab[y * w + x] == l {
                bin[y * w + x] = 255
                if x < x0 { x0 = x }
                if x > x1 { x1 = x }
                if y < y0 { y0 = y }
                if y > y1 { y1 = y }
            }
        }
        if x1 < 0 || y1 < 0 { return nil }

        var a = [UInt8](repeating: 0, count: w * h)
        let ya = max(0, y0 - 2), yb = min(h - 1, y1 + 2)
        let xa = max(0, x0 - 2), xb = min(w - 1, x1 + 2)
        for y in ya...yb {
            for x in xa...xb {
                var s = 0
                for dy in -1...1 {
                    for dx in -1...1 {
                        let yy = min(h - 1, max(0, y + dy))
                        let xx = min(w - 1, max(0, x + dx))
                        s += Int(bin[yy * w + xx])
                    }
                }
                a[y * w + x] = UInt8(s / 9)
            }
        }

        let W = CGFloat(w), H = CGFloat(h)
        let rect = CGRect(x: CGFloat(x0) / W, y: 1 - CGFloat(y1 + 1) / H,
                          width: CGFloat(x1 + 1 - x0) / W, height: CGFloat(y1 + 1 - y0) / H)
        return ObjInst(w: w, h: h, alpha: a, rect: rect, bx0: xa, bx1: xb, by0: ya, by1: yb)
    }

    static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let i = a.intersection(b)
        if i.isNull || i.isEmpty { return 0 }
        let inter = i.width * i.height
        let u = a.width * a.height + b.width * b.height - inter
        return u > 0 ? inter / u : 0
    }

    /// Какой из найденных объектов соответствует рамке трекера.
    static func match(_ insts: [ObjInst], rect r: CGRect) -> ObjInst? {
        var best: ObjInst?
        var bestV: CGFloat = 0.15
        for i in insts {
            let v = iou(i.rect, r)
            if v > bestV {
                bestV = v
                best = i
            }
        }
        return best
    }

    /// Объект под пальцем. nx, ny: 0...1, начало сверху слева.
    static func at(_ insts: [ObjInst], nx: Double, ny: Double) -> ObjInst? {
        var best: ObjInst?
        var bestArea = CGFloat.greatestFiniteMagnitude
        for o in insts {
            let x = min(o.w - 1, max(0, Int(nx * Double(o.w))))
            let y = min(o.h - 1, max(0, Int(ny * Double(o.h))))
            if o.alpha[y * o.w + x] >= 128 {
                let area = o.rect.width * o.rect.height
                if area < bestArea {
                    bestArea = area
                    best = o
                }
            }
        }
        return best
    }

    /// Контур маски (marching squares с интерполяцией), в координатах CGContext с началом снизу слева.
    static func contour(_ o: ObjInst, W: CGFloat, H: CGFloat) -> CGPath {
        let path = CGMutablePath()
        let w = o.w, h = o.h
        let iso = 127.5
        let xa = max(0, o.bx0), xb = min(w - 2, o.bx1)
        let ya = max(0, o.by0), yb = min(h - 2, o.by1)
        if xa > xb || ya > yb { return path }

        func v(_ x: Int, _ y: Int) -> Double { Double(o.alpha[y * w + x]) }
        func pt(_ x: Double, _ y: Double) -> CGPoint {
            CGPoint(x: (x + 0.5) / Double(w) * Double(W),
                    y: (1 - (y + 0.5) / Double(h)) * Double(H))
        }
        func lerp(_ p: Double, _ q: Double) -> Double {
            let d = q - p
            return abs(d) < 1e-6 ? 0.5 : (iso - p) / d
        }
        func seg(_ p: CGPoint, _ q: CGPoint) {
            path.move(to: p)
            path.addLine(to: q)
        }

        for y in ya...yb {
            for x in xa...xb {
                let a = v(x, y), b = v(x + 1, y), c = v(x + 1, y + 1), d = v(x, y + 1)
                var idx = 0
                if a >= iso { idx |= 8 }
                if b >= iso { idx |= 4 }
                if c >= iso { idx |= 2 }
                if d >= iso { idx |= 1 }
                if idx == 0 || idx == 15 { continue }

                let fx = Double(x), fy = Double(y)
                let top = pt(fx + lerp(a, b), fy)
                let right = pt(fx + 1, fy + lerp(b, c))
                let bottom = pt(fx + lerp(d, c), fy + 1)
                let left = pt(fx, fy + lerp(a, d))

                switch idx {
                case 1, 14: seg(left, bottom)
                case 2, 13: seg(bottom, right)
                case 3, 12: seg(left, right)
                case 4, 11: seg(top, right)
                case 6, 9: seg(top, bottom)
                case 7, 8: seg(top, left)
                case 5:
                    seg(top, right)
                    seg(left, bottom)
                case 10:
                    seg(top, left)
                    seg(bottom, right)
                default: break
                }
            }
        }
        return path
    }
}

// MARK: - Отрисовка

enum ObjectFX {
    // Последний разобранный кадр для превью (его заполняет редактор).
    static var insts: [ObjInst] = []
    static var base: CGImage?
    static var busy = false

    static func wants(_ fx: BoxFX) -> Bool {
        fx.has(.contour) || fx.has(.textOn) || fx.has(.textBehind)
    }

    private static func isObj(_ it: DrawItem) -> Bool { wants(it.fx) }

    /// Рисует эффекты по маске. Возвращает рамки, которые ещё надо нарисовать обычным путём:
    /// у объектов, найденных по маске, прямоугольник и подпись убираются (кроме выбранной рамки).
    static func apply(_ ctx: CGContext, _ W: CGFloat, _ H: CGFloat, _ items: [DrawItem],
                      _ insts: [ObjInst], _ base: CGImage?) -> [DrawItem] {
        var hit: [(item: DrawItem, inst: ObjInst)] = []
        var rest: [DrawItem] = []
        for it in items {
            if isObj(it), let o = ObjectSeg.match(insts, rect: it.rect) {
                hit.append((it, o))
                if it.selected { rest.append(it) }
            } else {
                rest.append(it)
            }
        }
        if hit.isEmpty { return rest }

        let full = CGRect(x: 0, y: 0, width: W, height: H)

        // 1. текст за объектом: сначала буквы, потом поверх вырезанный объект
        for h in hit where h.item.fx.has(.textBehind) {
            let r = h.item.rect
            let tw = max(r.width * 1.6, 0.4) * W
            drawText(ctx, h.item.label, cx: r.midX * W, cy: r.midY * H,
                     targetW: tw, maxH: r.height * H * 0.7, color: h.item.textColor.cg)
        }
        if let b = base {
            for h in hit where h.item.fx.has(.textBehind) {
                masked(ctx, h.inst, full) { ctx.draw(b, in: full) }
            }
        }

        // 2. текст на объекте: виден только там, где объект
        for h in hit where h.item.fx.has(.textOn) {
            let r = h.item.rect
            drawText(ctx, h.item.label, cx: r.midX * W, cy: r.midY * H,
                     targetW: r.width * W * 0.9, maxH: r.height * H * 0.5, color: h.item.textColor.cg, clip: (h.inst, full))
        }

        // 3. контур и лёгкая заливка по маске
        for h in hit where h.item.fx.has(.contour) {
            let c = h.item.color
            masked(ctx, h.inst, full) {
                ctx.setFillColor(Overlay.col(c, 0.16))
                ctx.fill(full)
            }
            let path = ObjectSeg.contour(h.inst, W: W, H: H)
            let lw = max(1.5, W * 0.004)
            ctx.saveGState()
            ctx.setLineCap(.round)
            ctx.setLineJoin(.round)
            ctx.addPath(path)
            ctx.setStrokeColor(Overlay.col(c, 0.3))
            ctx.setLineWidth(lw * 4.5)
            ctx.strokePath()
            ctx.addPath(path)
            ctx.setStrokeColor(Overlay.col(c, 1))
            ctx.setLineWidth(lw * 1.8)
            ctx.strokePath()
            ctx.restoreGState()
        }
        return rest
    }

    /// Всё, что нарисует draw, остаётся только там, где маска.
    private static func masked(_ ctx: CGContext, _ o: ObjInst, _ full: CGRect, _ draw: () -> Void) {
        guard let m = o.maskImage() else { return }
        ctx.saveGState()
        ctx.beginTransparencyLayer(auxiliaryInfo: nil)
        draw()
        ctx.setBlendMode(.destinationIn)
        ctx.interpolationQuality = .high
        ctx.draw(m, in: full)
        ctx.setBlendMode(.normal)
        ctx.endTransparencyLayer()
        ctx.restoreGState()
    }

    private static func drawText(_ ctx: CGContext, _ text: String, cx: CGFloat, cy: CGFloat,
                                 targetW: CGFloat, maxH: CGFloat, color: CGColor,
                                 clip: (ObjInst, CGRect)? = nil) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.isEmpty { return }

        func makeLine(_ s: CGFloat) -> CTLine? {
            let font = CTFontCreateWithName("HelveticaNeue-CondensedBold" as CFString, s, nil)
            let attrs: [CFString: Any] = [kCTFontAttributeName: font, kCTForegroundColorAttributeName: color]
            guard let a = CFAttributedStringCreate(nil, t as CFString, attrs as CFDictionary) else { return nil }
            return CTLineCreateWithAttributedString(a)
        }

        guard let l0 = makeLine(100) else { return }
        let b0 = CTLineGetBoundsWithOptions(l0, [])
        if b0.width < 1 { return }
        let size = max(8, min(maxH, 100 * targetW / b0.width))
        guard let l = makeLine(size) else { return }
        let b = CTLineGetBoundsWithOptions(l, [])

        let draw = {
            ctx.saveGState()
            ctx.textMatrix = .identity
            ctx.textPosition = CGPoint(x: cx - b.width / 2 - b.minX, y: cy - b.midY)
            CTLineDraw(l, ctx)
            ctx.restoreGState()
        }
        if let c = clip {
            masked(ctx, c.0, c.1, draw)
        } else {
            draw()
        }
    }

    // MARK: Экспорт: рисуем прямо в кадр

    static func draw(on pb: CVPixelBuffer, items: inout [DrawItem], time: Double) {
        guard items.contains(where: { isObj($0) }) else { return }
        let insts = ObjectSeg.instances(pixelBuffer: pb)

        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let ptr = CVPixelBufferGetBaseAddress(pb) else { return }

        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let info = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = CGContext(
            data: ptr, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info
        ) else { return }

        let clean = ctx.makeImage()
        items = apply(ctx, CGFloat(w), CGFloat(h), items, insts, clean)
    }

    // MARK: Превью: прозрачная картинка поверх плеера

    static func image(items: [DrawItem], size: CGSize, time: Double, base pixelBase: CGImage?) -> UIImage? {
        guard !items.isEmpty, size.width > 4, size.height > 4 else { return nil }
        let f = UIGraphicsImageRendererFormat()
        f.scale = 1.5
        f.opaque = false
        let renderer = UIGraphicsImageRenderer(size: size, format: f)
        return renderer.image { rc in
            let cg = rc.cgContext
            cg.translateBy(x: 0, y: size.height)
            cg.scaleBy(x: 1, y: -1)
            let rest = apply(cg, size.width, size.height, items, ObjectFX.insts, ObjectFX.base)
            Overlay.render(in: cg, W: size.width, H: size.height, items: rest, time: time, base: pixelBase)
        }
    }
}

// MARK: - Редактор

extension Editor {
    /// Разобрать текущий кадр (на паузе), чтобы превью показало точный контур.
    func objRefresh() {
        guard boxes.contains(where: { ObjectFX.wants($0.fx) }) else {
            if !ObjectFX.insts.isEmpty {
                ObjectFX.insts = []
                ObjectFX.base = nil
            }
            return
        }
        if isPlaying { return }
        let t = currentTime
        Task { @MainActor in
            guard let cg = await self.frameImage(at: t) else { return }
            let r = await Task.detached { ObjectSeg.instances(image: cg) }.value
            if abs(self.currentTime - t) > 0.002 { return }
            ObjectFX.insts = r
            ObjectFX.base = cg
            self.objectWillChange.send()
        }
    }

    /// Во время воспроизведения: берём кадр из плеера (приблизительно, с небольшим отставанием).
    func objTick() {
        guard isPlaying, !ObjectFX.busy,
              boxes.contains(where: { ObjectFX.wants($0.fx) }) else { return }
        let it = player.currentTime()
        guard let pb = videoOutput.copyPixelBuffer(forItemTime: it, itemTimeForDisplay: nil) else { return }
        ObjectFX.busy = true
        Task.detached {
            let r = ObjectSeg.instances(pixelBuffer: pb)
            var made: CGImage?
            VTCreateCGImageFromCVPixelBuffer(pb, options: nil, imageOut: &made)
            let image = made
            await MainActor.run {
                ObjectFX.insts = r
                ObjectFX.base = image
                ObjectFX.busy = false
                self.objectWillChange.send()
            }
        }
    }

    /// Выбранной рамке включить точный контур (после выбора объекта).
    func markObject() {
        if let id = selectedBoxID, let i = boxes.firstIndex(where: { $0.id == id }) {
            boxes[i].fx.on.insert(.contour)
        }
        objRefresh()
    }
}
