#include "dmp/utility/utility.h"

namespace dmp {

int file_type(const char* object_path) {
  struct stat s;
  if (stat(object_path, &s) == 0) {
    if (s.st_mode & S_IFDIR) {
      return _DIR_;  // it's a directory
    } else if (s.st_mode & S_IFREG) {
      return _REG_;  // it's a file
    } else {
      return 0;  // something else
    }
  } else {
    return -1;  // error
  }
}

uint countNumberedDirectoriesUnderDirectory(const char* dir_path) {
  char var_dir_path[1000];
  uint count_numbered_directories = 1;
  sprintf(var_dir_path, "%s/%d/", dir_path, count_numbered_directories);

  while (file_type(var_dir_path) == _DIR_)  // while directory exists
  {
    count_numbered_directories++;
    sprintf(var_dir_path, "%s/%d/", dir_path, count_numbered_directories);
  }
  count_numbered_directories--;
  return (count_numbered_directories);
}

uint countNumberedFilesUnderDirectory(const char* dir_path) {
  char var_file_path[1000];
  uint count_numbered_files = 1;
  sprintf(var_file_path, "%s/%d.txt", dir_path, count_numbered_files);

  while (file_type(var_file_path) == _REG_)  // while file exists
  {
    count_numbered_files++;
    sprintf(var_file_path, "%s/%d.txt", dir_path, count_numbered_files);
  }
  count_numbered_files--;
  return (count_numbered_files);
}

bool createDirIfNotExistYet(const char* dir_path) {
  if (strcmp(dir_path, "") == 0)  // if the dir_path is a NULL string ...
  {
    return false;
  }

  if (file_type(dir_path) !=
      _DIR_)  // if the dir_path directory doesn't exist yet ...
  {
    return (std::filesystem::create_directories(dir_path));  // then create it
  } else {
    return true;
  }
}

Vector3 computeClosestPointOnSphereSurface(const Vector3& evaluation_point,
                                           const Vector3& sphere_center_point,
                                           const double& sphere_radius) {
  // sphctr_to_evalpt_vector is a vector originating from the sphere center,
  // and going to the evaluation point:
  Vector3 sphctr_to_evalpt_vector = evaluation_point - sphere_center_point;

  // sphctr_to_evalpt_distance is the distance between the sphere center and
  // the evaluation point:
  double sphctr_to_evalpt_distance = sphctr_to_evalpt_vector.norm();

  // closestpt_on_sphsurf is a point on the sphere's surface,
  // which is closest to the evaluation point:
  Vector3 closestpt_on_sphsurf =
      sphere_center_point +
      ((sphctr_to_evalpt_vector / sphctr_to_evalpt_distance) * sphere_radius);

  return (closestpt_on_sphsurf);
}

}  // namespace dmp
