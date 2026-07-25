// Minimal UnityEngine surface so the real ParkourBox.cs / ParkourTrackGenerator.cs
// compile and run outside the editor. Only what those two files touch.
using System;

namespace UnityEngine
{
    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public static Vector3 zero => new Vector3(0, 0, 0);
        public static Vector3 one => new Vector3(1, 1, 1);
        public static Vector3 right => new Vector3(1, 0, 0);
        public static Vector3 up => new Vector3(0, 1, 0);
        public static Vector3 forward => new Vector3(0, 0, 1);
        public static Vector3 operator +(Vector3 a, Vector3 b) => new Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
        public static Vector3 operator -(Vector3 a, Vector3 b) => new Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
        public static Vector3 operator -(Vector3 a) => new Vector3(-a.x, -a.y, -a.z);
        public static Vector3 operator *(Vector3 a, float s) => new Vector3(a.x * s, a.y * s, a.z * s);
        public static Vector3 operator *(float s, Vector3 a) => a * s;
        public static Vector3 operator /(Vector3 a, float s) => new Vector3(a.x / s, a.y / s, a.z / s);
        public static float Dot(Vector3 a, Vector3 b) => a.x * b.x + a.y * b.y + a.z * b.z;
        public static Vector3 Scale(Vector3 a, Vector3 b) => new Vector3(a.x * b.x, a.y * b.y, a.z * b.z);
        public static bool operator ==(Vector3 a, Vector3 b) => a.x == b.x && a.y == b.y && a.z == b.z;
        public static bool operator !=(Vector3 a, Vector3 b) => !(a == b);
        public override bool Equals(object o) => o is Vector3 v && this == v;
        public override int GetHashCode() => x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
        public override string ToString() => $"({x:F2},{y:F2},{z:F2})";
    }

    public struct Quaternion
    {
        public float x, y, z, w;
        public Quaternion(float x, float y, float z, float w) { this.x = x; this.y = y; this.z = z; this.w = w; }
        public static Quaternion identity => new Quaternion(0, 0, 0, 1);

        public static Quaternion Euler(float px, float py, float pz)
        {
            // Unity order: Z, then X, then Y (intrinsic), degrees.
            float cx = (float)Math.Cos(px * Mathf.Deg2Rad * 0.5f), sx = (float)Math.Sin(px * Mathf.Deg2Rad * 0.5f);
            float cy = (float)Math.Cos(py * Mathf.Deg2Rad * 0.5f), sy = (float)Math.Sin(py * Mathf.Deg2Rad * 0.5f);
            float cz = (float)Math.Cos(pz * Mathf.Deg2Rad * 0.5f), sz = (float)Math.Sin(pz * Mathf.Deg2Rad * 0.5f);
            return new Quaternion(
                sx * cy * cz + cx * sy * sz,
                cx * sy * cz - sx * cy * sz,
                cx * cy * sz - sx * sy * cz,
                cx * cy * cz + sx * sy * sz);
        }

        public static bool operator ==(Quaternion a, Quaternion b)
            => a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
        public static bool operator !=(Quaternion a, Quaternion b) => !(a == b);
        public override bool Equals(object o) => o is Quaternion q && this == q;
        public override int GetHashCode()
            => x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();

        public static Vector3 operator *(Quaternion q, Vector3 v)
        {
            float nx = q.x * 2f, ny = q.y * 2f, nz = q.z * 2f;
            float xx = q.x * nx, yy = q.y * ny, zz = q.z * nz;
            float xy = q.x * ny, xz = q.x * nz, yz = q.y * nz;
            float wx = q.w * nx, wy = q.w * ny, wz = q.w * nz;
            return new Vector3(
                (1f - (yy + zz)) * v.x + (xy - wz) * v.y + (xz + wy) * v.z,
                (xy + wz) * v.x + (1f - (xx + zz)) * v.y + (yz - wx) * v.z,
                (xz - wy) * v.x + (yz + wx) * v.y + (1f - (xx + yy)) * v.z);
        }
    }

    public static class Mathf
    {
        public const float Deg2Rad = 0.0174532924f;
        public static float Clamp(float v, float a, float b) => v < a ? a : (v > b ? b : v);
        public static int Clamp(int v, int a, int b) => v < a ? a : (v > b ? b : v);
        public static float Clamp01(float v) => Clamp(v, 0f, 1f);
        public static float Lerp(float a, float b, float t) => a + (b - a) * Clamp01(t);
        public static float Abs(float v) => Math.Abs(v);
        public static float Max(float a, float b) => Math.Max(a, b);
        public static int Max(int a, int b) => Math.Max(a, b);
        public static float Min(float a, float b) => Math.Min(a, b);
        public static int Min(int a, int b) => Math.Min(a, b);
        public static int RoundToInt(float v) => (int)Math.Round(v, MidpointRounding.AwayFromZero);
        public static int CeilToInt(float v) => (int)Math.Ceiling(v);
        public static int FloorToInt(float v) => (int)Math.Floor(v);
        public static float Tan(float v) => (float)Math.Tan(v);
        public static float Atan2(float a, float b) => (float)Math.Atan2(a, b);
        public static float Sqrt(float v) => (float)Math.Sqrt(v);
    }

    public static class Debug
    {
        public static void LogError(object m) => Console.Error.WriteLine($"[error] {m}");
        public static void LogWarning(object m) => Console.Error.WriteLine($"[warn] {m}");
        public static void Log(object m) => Console.WriteLine(m);
    }

    public class Object { public string name = ""; }
    // Null here, which is what the harness relies on: it drives the generator with
    // no GameObject behind it, so the track frame is world space and the transform
    // guard in Generate() is skipped.
    public class Component : Object { public Transform transform; }
    public class Behaviour : Component { }
    public class MonoBehaviour : Behaviour
    {
        public static GameObject Instantiate(GameObject o, Transform parent) => new GameObject();
    }
    public class Material : Object { }
    public class MeshRenderer : Component { public Material sharedMaterial; }
    // No field initialisers here: Transform.gameObject and GameObject.transform
    // referring to each other would recurse forever on construction.
    public class Transform : Component
    {
        public Vector3 localScale;
        public Vector3 position;
        public Quaternion rotation;
        public Vector3 lossyScale;
        public GameObject gameObject;
        public void SetPositionAndRotation(Vector3 p, Quaternion r) { }
        public void SetLocalPositionAndRotation(Vector3 p, Quaternion r) { }
        public T GetComponent<T>() where T : class => null;
    }
    public class GameObject : Object
    {
        public Transform transform;
        public void SetActive(bool v) { }
    }

    [AttributeUsage(AttributeTargets.Field)] public class TooltipAttribute : Attribute { public TooltipAttribute(string s) { } }
    [AttributeUsage(AttributeTargets.Field)] public class HeaderAttribute : Attribute { public HeaderAttribute(string s) { } }
    [AttributeUsage(AttributeTargets.Field)] public class SerializeFieldAttribute : Attribute { }
}
