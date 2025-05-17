PShape baseShape, shoulderShape, upperArmShape, lowerArmShape, endEffectorShape;
float rotationX, rotationY;
float positionX = 1, positionY = 50, positionZ = 50;

float previousMillis, globalTime, globalSpeed = 4;

float moveStep = 0.5;
boolean isGrabbing = false, hasCollisionGlobal;

String[] movementCommands;
String[] stateCommands;

class Cube {
  PVector currentPosition;
  PVector previousPosition;
  boolean hasCollided;

  float width = 20, height = 20, depth = 20; // Dimensiones del cubo
  float initialRotation;
  int floorLevel = -68;
  float hitBoxRadius = (width + height + depth) / 3;

  // Constructor: establece posición inicial y rotación
  Cube(float startX, float topY, float startZ, float startRotation) {
    float adjustedY = topY + floorLevel + height / 2;
    currentPosition = new PVector(startX, adjustedY, startZ);
    initialRotation = startRotation;
  }

  // Actualiza la posición del cubo según coordenadas del robot
  void updatePosition(float robotX, float robotY, float robotZ) {
    float xProc = -robotY;
    float yProc = -robotZ;
    float zProc = -robotX;
    currentPosition.set(xProc, yProc, zProc);
  }

  // Verifica colisión entre el end-effector y el cubo
  boolean checkCollisionWithEffector(float robotX, float robotY, float robotZ) {
    float xProc = -robotY;
    float yProc = -robotZ;
    float zProc = -robotX;
    PVector distanceVec = PVector.sub(currentPosition, new PVector(xProc, yProc, zProc));
    return distanceVec.mag() < hitBoxRadius;
  }

  // Verifica colisión entre dos cubos y restaura posición si colisionan
  boolean checkCollisionWithCube(Cube otherCube) {
    PVector distanceVec = PVector.sub(otherCube.currentPosition, currentPosition);
    if (distanceVec.mag() <= hitBoxRadius) {
      currentPosition = previousPosition.copy();
      hasCollided = true;
    } else {
      hasCollided = false;
    }
    return hasCollided;
  }

  // Simula la caída si no hay colisión ni agarre
  void applyGravity() {
    if (!hasCollided && !isGrabbing && currentPosition.y > floorLevel + height / 2) {
      currentPosition.y -= 0.1 * (previousMillis - globalTime);
    }
  }
  
  // Dibuja el cubo y actualiza la posición previa
  void display() {
    if (currentPosition.y < floorLevel + height / 2) {
      currentPosition.y = floorLevel + height / 2;
    }
    fill(120, 120, 60);
    pushMatrix();
    translate(currentPosition.x, currentPosition.y, currentPosition.z);
    rotateY(initialRotation);
    box(width, height, depth);
    popMatrix();
    previousPosition = currentPosition.copy();
  }
}

// Ajusta la posición del end-effector según comandos de movimiento y estado de agarre
void updateRobotPosition() {
  if (movementCommands != null) {
    for (String cmd : movementCommands) {
      switch(cmd) {
        case "Abajo":    if (positionZ < maxZ) positionZ += moveStep; break;
        case "Arriba":   if (positionZ > minZ) positionZ -= moveStep; break;
        case "Izquierda":if (positionX > minX) positionX -= moveStep; break;
        case "Derecha":  if (positionX < maxX) positionX += moveStep; break;
        case "Atras":    if (positionY < maxY) positionY += moveStep; break;
        case "Adelante": if (positionY > minY) positionY -= moveStep; break;
      }
    }
  }
  if (stateCommands != null) {
    for (String st : stateCommands) {
      switch(st) {
        case "Abierta": isGrabbing = false; break;
        case "Cerrada": isGrabbing = true; break;
      }
    }
  }
}

// Longitudes de los segmentos del brazo
float shoulderLength = 50;
float forearmLength = 70;

float angleAlpha, angleBeta, angleGamma;

// Límites de movimiento del effector
float minX = -68, maxX = 68;
float minY = -68, maxY = 68;
float minZ = -68, maxZ = 62;

// Calcula la cinemática inversa del brazo
void calculateInverseKinematics() {
  float X = positionX;
  float Y = positionY;
  float Z = positionZ;
  float L = sqrt(Y*Y + X*X);
  float dist = sqrt(Z*Z + L*L);

  angleAlpha = PI/2 - (atan2(L, Z) + acos((forearmLength*forearmLength - shoulderLength*shoulderLength - dist*dist) / (-2*shoulderLength*dist)));
  angleBeta  = -PI + acos((dist*dist - forearmLength*forearmLength - shoulderLength*shoulderLength) / (-2*shoulderLength*forearmLength));
  angleGamma = atan2(Y, X);
}

// Arreglo de cubos en escena
Cube[] cubes = {
  new Cube(65, 0, -65, 0),
  new Cube(65, 40, -65, 0),
  new Cube(3, 0, -35, 0),
  new Cube(45, 0, 20, 0),
  new Cube(39, 0, -29, 0)
};

int[] collisionCounts = new int[cubes.length];
boolean[] cubeCrashed = new boolean[cubes.length];

// Actualiza el tiempo para simulación de gravedad
void updateTime() {
  globalTime += ((float)millis()/1000 - previousMillis) * (globalSpeed/4);
  if (globalTime >= 4) globalTime = 0;
  previousMillis = (float)millis()/1000;
}

// Actualiza posición y cinemática antes de dibujar
void updateKinematics() {
  updateRobotPosition();
  calculateInverseKinematics();
}

// Coordenadas de puntos en esfera (no usadas actualmente)
float[] xSpherePoints = new float[99];
float[] ySpherePoints = new float[99];
float[] zSpherePoints = new float[99];

void setup() {
  size(1200, 800, OPENGL);
  baseShape      = loadShape("r5.obj");
  shoulderShape  = loadShape("r1.obj");
  upperArmShape  = loadShape("r2.obj");
  lowerArmShape  = loadShape("r3.obj");
  endEffectorShape = loadShape("r4.obj");
  shoulderShape.disableStyle();
  upperArmShape.disableStyle();
  lowerArmShape.disableStyle();
}

void draw() {
  movementCommands = loadStrings("handDir.txt");
  stateCommands    = loadStrings("hand_state.txt");

  updateKinematics();
  updateTime();

  background(31);
  smooth(); lights();
  directionalLight(51, 102, 126, -1, 0, 0);
  noStroke(); fill(255);
  translate(width/2, height/2);

  rotateX(rotationX);
  rotateY(-rotationY);
  scale(-4);

  // Dibuja el suelo
  pushMatrix();
    translate(0, -68, 0);
    box(300, 2, 300);
  popMatrix();

  // Dibuja el brazo robótico
  pushMatrix();
    fill(150, 0, 150);
    translate(0, -40, 0);
    shape(baseShape);

    translate(0, 4, 0);
    rotateY(angleGamma);
    shape(shoulderShape);

    translate(0, 25, 0);
    rotateY(PI);
    rotateX(angleAlpha);
    shape(upperArmShape);

    translate(0, 0, 50);
    rotateY(PI);
    rotateX(angleBeta);
    shape(lowerArmShape);

    translate(0, 0, -50);
    rotateY(PI);
    shape(endEffectorShape);
  popMatrix();

  // Dibuja y actualiza cubos
  for (int i = 0; i < cubes.length; i++) {
    cubes[i].display();
    if (cubes[i].checkCollisionWithEffector(positionX, positionY, positionZ) && isGrabbing) {
      cubes[i].updatePosition(positionX, positionY, positionZ);
    }
    collisionCounts[i] = 0;
    for (int j = 0; j < cubes.length; j++) {
      if (j != i) {
        boolean collided = cubes[i].checkCollisionWithCube(cubes[j]);
        if (collided) {
          cubeCrashed[i] = true;
          collisionCounts[i]++;
        }
      }
    }
    cubeCrashed[i] = (collisionCounts[i] > 0) ? true : false;
    if (!cubeCrashed[i]) {
      cubes[i].applyGravity();
    }
  }
}

void mouseDragged() {
  rotationY -= (mouseX - pmouseX) * 0.01;
  rotationX -= (mouseY - pmouseY) * 0.01;
}
