//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, torus, mobius
// Camera: perspective
// Light: point
//=============================================================================================
#include "framework.h"

const int tessellationLevel = 40;
bool gameOver = false;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 20;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}

	void Animate(float t) { }
};

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos;

	void Animate(float t) {	}
};

float randf() {
	return rand() / (float)RAND_MAX;
}
//---------------------------
class CheckerBoardTexture : public Texture {
	//---------------------------
public:
	CheckerBoardTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};
class RedTexture : public Texture {
	//---------------------------
public:
	RedTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 red(1, 0, 0, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = red;
		}
		create(width, height, image, GL_NEAREST);
	}
};
class StripesTexture : public Texture {
	//---------------------------
public:
	StripesTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 red(1, 0, 0, 1);
		const vec4 green(0, 1, 0, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			if (x%2==0)
				image[y * width + x] = red;
			else
				image[y * width + x] = green;
		}
		create(width, height, image, GL_NEAREST);
	}
}; 
class SkyTexture : public Texture {
	//---------------------------
public:
	SkyTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 lightBlue(0.5, 0.8, 1, 1);
		const vec4 darkBlue(0, 0, 0.7, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			if (x % 2 == 0)
				image[y * width + x] = lightBlue;
			else
				image[y * width + x] = darkBlue;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};


//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};


//---------------------------
struct VertexData {
	//---------------------------
	vec3 position, normal;
	vec2 texcoord;
};

//---------------------------
class Geometry {
	//---------------------------
protected:
	unsigned int vao, vbo;        // vertex array object
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};
class triangle {
public :
	VertexData data[3];
	triangle(vec3 _a, vec3 _b, vec3 _c) {
		vec3 normal = normalize(cross(_a - _b, _a - _c));
		data[0].position = _a;
		data[0].normal = normal;
		data[0].texcoord = { 0,0 };
		data[1].position = _b;
		data[1].normal = normal;
		data[1].texcoord = { 0,0 };
		data[2].position = _c;
		data[2].normal = normal;
		data[2].texcoord = { 0,0 };
	}
};
class Fractal : public Geometry {
public:
	int depth;
	int triangleCount = 0;
	Fractal(int depth, float h) { create(depth, h); }
	void create(int depth, float h) {
		std::vector<triangle> triangles;
		triangles.reserve(4 * pow(7, depth+1));
		vec3 pos1 = { 1,1,1 };
		vec3 pos2 = { 1,-1,-1 };
		vec3 pos3 = { -1,1,-1 };
		vec3 pos4 = { -1,-1,1 };

		triangles.push_back({pos1,pos2,pos3 });
		triangles.push_back({pos1, pos4, pos2});
		triangles.push_back({pos1,pos3,pos4});
		triangles.push_back({pos3,pos2,pos4});
		for (auto& i : triangles) {
			rec(triangles,i, depth,h);
		}
		triangleCount = triangles.size();
	
		glBufferData(GL_ARRAY_BUFFER, triangles.size()*sizeof(triangle), &triangles[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 1 = NORMAL

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	void rec(std::vector<triangle>& triangles, triangle tri, int depth, float h) {

		vec3 a = tri.data[0].position;
		vec3 b = tri.data[1].position;
		vec3 c = tri.data[2].position;
		vec3 aab = (a  + b) / 2.0;
		vec3 aac = (a  + c) / 2.0;
		vec3 bbc = (b + c) / 2.0;
		vec3 top = (a + b + c) / 3.0 + tri.data[0].normal * h;
		triangles.push_back({ aab,bbc, top });
		triangles.push_back({ bbc,aac, top });
		triangles.push_back({ aac,aab, top });
		if (depth != 0) {
			rec(triangles, { aab,bbc, top }, depth - 1, h/3);
			rec(triangles, { bbc,aac, top }, depth - 1, h/3);
			rec(triangles, { aac,aab, top }, depth - 1, h/3);
			rec(triangles, { a,aab, aac }, depth - 1, h/3);
			rec(triangles, { b,bbc, aab }, depth - 1, h/3);
			rec(triangles, { c,aac, bbc }, depth - 1, h/3);
		}

	}
	void Draw(){
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES,0,triangleCount*3);
	}
};
//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
protected:
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual VertexData GenVertexData(float u, float v, float tend = 0) = 0;
	// virtual VertexData GenVertexData(float u, float v, float t) = 0;

	void create(int N = tessellationLevel, int M = tessellationLevel, float tend = 0) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N, tend));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N,tend));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};
//---------------------------
struct Clifford {
	//---------------------------
	float f, d;
	Clifford(float f0 = 0, float d0 = 0) { f = f0, d = d0; }
	Clifford operator+(Clifford r) { return Clifford(f + r.f, d + r.d); }
	Clifford operator-(Clifford r) { return Clifford(f - r.f, d - r.d); }
	Clifford operator*(Clifford r) { return Clifford(f * r.f, f * r.d + d * r.f); }
	Clifford operator/(Clifford r) {
		float l = r.f * r.f;
		return (*this) * Clifford(r.f / l, -r.d / l);
	}
};

Clifford T(float t) { return Clifford(t, 1); }
Clifford Sin(Clifford g) { return Clifford(sin(g.f), cos(g.f) * g.d); }
Clifford Cos(Clifford g) { return Clifford(cos(g.f), -sin(g.f) * g.d); }
Clifford Tan(Clifford g) { return Sin(g) / Cos(g); }
Clifford Log(Clifford g) { return Clifford(logf(g.f), 1 / g.f * g.d); }
Clifford Exp(Clifford g) { return Clifford(expf(g.f), expf(g.f) * g.d); }
Clifford Pow(Clifford g, float n) { return Clifford(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }
Clifford Cosh(Clifford g) { return Clifford(cosh(g.f), sinh(g.f) * g.d); }
Clifford Tanh(Clifford g) { return Clifford(sinh(g.f) / cosh(g.f), (1.0f / (cosh(g.f) * cosh(g.f)) * g.d)); }
vec4 normalize(vec4 v) {
	float d = sqrt(dot(v, v));
	return vec4(v.x / d, v.y / d, v.z / d, v.w / d);
}
class Sphere : public ParamSurface {
	//---------------------------
public:
	Sphere() { create(); }
	VertexData GenVertexData(float u, float v, float tend) {

		VertexData vd;
		Clifford U(u * 2.0f * M_PI,1);
		Clifford V(v * M_PI,0);
		Clifford X = Cos(U) * Sin(V);
		Clifford Y = Sin(U) * Sin(V);
		Clifford Z = Cos(V);
		vd.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU = vec3(X.d, Y.d, Z.d);
		U.d = 0; V.d = 1;
		X = Cos(U) * Sin(V);
		Y = Sin(U) * Sin(V);
		Z = Cos(V);

		vec3 drdV = vec3(X.d, Y.d, Z.d);
		vd.normal = normalize(cross(drdU, drdV));
		vd.texcoord = vec2(u, v);

		/*vd.position = vd.normal = vec3(cosf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
			sinf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
			cosf(v * (float)M_PI));
		vd.texcoord = vec2(u, v);*/
		return vd;
	}
};
class Tractricoid : public ParamSurface {
	//---------------------------
public:
	Tractricoid(float t) { create(tessellationLevel, tessellationLevel, t); }
	VertexData GenVertexData(float u, float v, float tend = 0) {

		VertexData vd;
		Clifford U(u * 4.0, 1);
		Clifford V(v * 2 * M_PI,0);
		Clifford X = Cos(V) / Cosh(U);
		Clifford Y = Sin(V) / Cosh(U);
		Clifford Z = U - Tanh(U);
		vd.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU = vec3(X.d, Y.d, Z.d);

		U.d = 0; V.d = 1;
		X = Cos(V) / Cosh(U);
		Y = Sin(V) / Cosh(U);
		Z = U - Tanh(U);
		
		vec3 drdV = vec3(X.d, Y.d, Z.d);
		vd.normal = normalize(cross(drdU, drdV));
		vd.texcoord = vec2(u, v);

		return vd;
	}
};
Clifford generateR(Clifford U, Clifford V, float t) {
	return Clifford((sin(t*2)+1)/8.0, 0) * Sin(U + t) * Sin(V * 10 + t) + (1- (sin(t) + 1) / 8.0);
}
class VirusBody : public ParamSurface {
public:
	VirusBody(float t) {
		create(tessellationLevel, tessellationLevel,t);
	}
	VertexData GenVertexData(float u, float v, float t) {

		VertexData vd;
		Clifford U(u * 2.0f * M_PI, 1);
		Clifford V(v * M_PI, 0);
		Clifford R = generateR(U, V, t);
		Clifford X = Cos(U) * Sin(V) *R;
		Clifford Y = Sin(U) * Sin(V) *R;
		Clifford Z = Cos(V)*R;
		vd.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU = vec3(X.d, Y.d, Z.d);
		U.d = 0; V.d = 1;
		R = generateR(U, V, t);
		X = Cos(U) * Sin(V) * R;
		Y = Sin(U) * Sin(V) * R;
		Z = Cos(V)*R;

		vec3 drdV = vec3(X.d, Y.d, Z.d);
		vd.normal = normalize(cross(drdU, drdV));
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class Object {
public:
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object() {}
	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 1, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}
	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	virtual void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { rotationAngle = tend * 0.1; }
};
class Virus : public Object {
public:
	std::vector<Object*> tractris;
	Virus(Shader* _shader, Material* _material, Texture* _texture) {
		scale = vec3(1, 1, 1);
		translation = vec3(0, 0, 0);
		rotationAxis = vec3(0, 0, 1);
		rotationAngle = 0;
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = new VirusBody(0);
		
		printf("%d\n", tractris.size());
		Animate(0, 0);
	}
	void Animate(float tstart, float tend) {
		translation = translation + vec3(randf() - 0.5, randf() - 0.5, randf() - 0.5)*0.1;
		delete geometry;
		geometry = new VirusBody(tend);
		// https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
		vec4 Q = normalize(vec4(cos(tend), sin(tend / 2), sin(tend / 3), sin(tend / 5)));
		rotationAngle = acos(Q.x) * 2.0f;
		float div = sin(rotationAngle / 2.0f);
		rotationAxis = vec3(Q.y / div, Q.z / div, Q.w / div);
	}
};
class Tracti : public Object {
public:
	float u, v;
	vec3 dir;
	Object* body;
	vec3 placeTranslate;
	Tracti(Shader* _shader, Material* _material, Texture* _texture, float _u, float _v, Object * _body) {
		u = _u;
		v = _v;
		dir = { 1,0,0 };
		scale = vec3(0.15, 0.15, 0.1);
		translation = vec3(0, 0, 0);
		placeTranslate = vec3(0, 0, 0);
		rotationAxis = vec3(0, 0, 1);
		rotationAngle = 0;
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = new Tractricoid(0);
		body = _body;
	}
	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		vec3 second = normalize(vec3(1,1,-(dir.x+dir.y)/(dir.z+0.01)));
		vec3 third = normalize(cross( dir, second));
		if (dot(cross(second, third), dir) < 0) {
			second = -second;
		}
		if (dot(cross(third, dir), second) < 0) {
			third = -third;
		}
		mat4 inv = { second.x, third.x, dir.x, 0,
			second.y, third.y, dir.y, 0,
			 second.z,third.z, dir.z, 0,
			0, 0, 0, 1 };
		mat4 rotator = mat4(
			second.x, second.y, second.z, 0,
			third.x, third.y, third.z, 0,
			dir.x, dir.y, dir.z,0,
			0,0,0,1 );
		
		M = ScaleMatrix(scale) *rotator *TranslateMatrix(translation) *RotationMatrix(rotationAngle, rotationAxis)*TranslateMatrix(placeTranslate) ;
		Minv = TranslateMatrix(placeTranslate)* RotationMatrix(-rotationAngle, rotationAxis)* TranslateMatrix(-translation)*inv * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}
	void Animate(float tstart, float tend) {
		Clifford U(u * 2.0f * M_PI, 1);
		Clifford V(v * M_PI, 0);
		Clifford R = generateR(U,V,tend);
		Clifford X = Cos(U) * Sin(V) * R;
		Clifford Y = Sin(U) * Sin(V) * R;
		Clifford Z = Cos(V) * R;
		translation = vec3(X.f, Y.f, Z.f);
		vec3 drdU = vec3(X.d, Y.d, Z.d);
		U.d = 0; V.d = 1;
		R = generateR(U,V,tend);
		X = Cos(U) * Sin(V) * R;
		Y = Sin(U) * Sin(V) * R;
		Z = Cos(V) * R;

		vec3 drdV = vec3(X.d, Y.d, Z.d);
		dir = normalize(cross(drdU, drdV));
		translation = vec3(X.f, Y.f, Z.f) -  dir*0.3;
		
		placeTranslate = body->translation;
		rotationAxis = body->rotationAxis;
		rotationAngle = body->rotationAngle;
	}

};
vec3 prob;
class Antibody : public Object {
public:
	float t;
	vec3 direction;
	float R;
	Virus* virus;
	Antibody(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) {
		scale = vec3(0.6, 0.6 ,0.6);
		translation = vec3(-5, -1, -7);
		rotationAxis = vec3(1, 1, 1);
		direction = vec3(0, 0, 0);
		prob = normalize(vec3(1, 1, 1));
		t = 0;
		rotationAngle = 0;
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}
	void Animate(float tstart, float tend) {
		float deltat = tend - tstart;
		if (t + deltat > 0.5) {
			t = 0;
			direction = normalize(vec3(randf() - 0.5, randf() - 0.5, randf() - 0.5) + prob);
			prob = normalize(vec3(randf() - 0.5, randf() - 0.5, randf() - 0.5));
		}
		t += deltat;
		rotationAngle = 0.8f * tend;
		delete geometry;
		int depth = (sin(tend * 2) + 1) * 2.0;
		float height = (sin(tend * 8.0) + 2.0) * 1.1;
		translation = translation + direction * deltat * 0.8;
		geometry = new Fractal(depth, height);
		R = height/3.0 + sqrt(1.00 / 9.00);
		vec3 dist = translation - virus->translation;
		if (dot(dist, dist) < (R + 1) * (R + 1))
		{
			gameOver = true;
		}
	}
};
//---------------------------
class Scene {
//---------------------------
	std::vector<Object*> objects;
	Camera camera; // 3D camera
	std::vector<Light> lights;
public:
	void Build() {
		// Shaders
		Shader* phongShader = new PhongShader();

		// Materials
		Material* material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		// Textures
		Texture* texture4x8 = new CheckerBoardTexture(4, 8);
		Texture* texture15x20 = new CheckerBoardTexture(15, 20);
		Texture* textureRed = new RedTexture(15, 20);
		Texture* textureStripes = new StripesTexture(16, 20);
		Texture* textureSky = new SkyTexture(160, 200);

		Geometry* sphere = new Sphere();

		// Create objects by setting up their vertex data on the GPU
		Virus* virus = new Virus(phongShader, material0, textureStripes);
		virus->translation = vec3(2, -1, -2);
		virus->rotationAxis = vec3(0, 1, 1);
		virus->scale = vec3(1.0f, 1.0f, 1.0f);
		objects.push_back(virus);

		Geometry* tractri = new Tractricoid(0);

		Object* universe = new Object(phongShader, material0, textureSky, sphere);
		universe->scale = vec3(10, 10, 10);
		universe->translation = vec3(0, 0, 0);
		universe->rotationAngle = M_PI;
		objects.push_back(universe);

		int limit = 10;
		for (int i = 0; i <= limit; i++) {
			int nextLimit = (int)((float)limit * (sin((float)i / limit * M_PI)));
			for (int j = 0; j <= nextLimit; j++) {
				float u = (float)j / (nextLimit + 1)+0.01;
				float v = (float)i / limit+0.01;

				Object* Tractri = new Tracti(phongShader, material0, textureRed, u, v, virus);

				objects.push_back(Tractri);
			}
		}
		Geometry* fractal = new Fractal(0,1);
		Antibody* fractalObject = new Antibody(phongShader, material0, textureRed, fractal);
		objects.push_back(fractalObject);
		fractalObject->virus = virus;
		// Camera
		camera.wEye = vec3(0, 0, 6);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		// Lights
		lights.resize(1);
		lights[0].wLightPos = vec4(5, 5, 4, 1);	// ideal point -> directional light source
		lights[0].La = vec3(4.0f, 4.0f, 4.2f);
		lights[0].Le = vec3(0.5, 0.5, 0.5);

	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		if (!gameOver) {
			camera.Animate(tend);
			for (unsigned int i = 0; i < lights.size(); i++) { lights[i].Animate(tend); }
			for (Object* obj : objects) obj->Animate(tstart, tend);
		}
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
		case 'x': prob = normalize(prob + vec3(0.1, 0, 0));
			break;
		case 'y': prob = normalize(prob + vec3(0, 0.1, 0));
			break;
		case 'z': prob = normalize(prob + vec3(0, 0, 0.1));
			break;
		case 'X': prob = normalize(prob + vec3(-0.1, 0, 0));
			break;
		case 'Y': prob = normalize(prob + vec3(0, -0.1, 0));
			break; 
		case 'Z': prob = normalize(prob + vec3(0, 0, -0.1));
			break;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}
